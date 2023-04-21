import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch
import sys
import numpy as np

import logging
lamorel_logger = logging.getLogger('lamorel_logger')

from .llms import HF_LLM
from .llms.updaters import BaseUpdater
from .llms.module_functions import BaseModuleFunction, ScoreModuleFunction
from .dispatcher import Dispatcher
from .utils import InstructionsEnum

from accelerate import Accelerator
accelerator = Accelerator()

class Server:
    def __init__(self, config, llm_index, llm_group, llm_master, rl_llm_group, rl_llm_group_size, custom_updater,
                 custom_module_functions, custom_model_initializer):
        assert dist.is_initialized(), "torch distributed must be used!"
        self._index = llm_index  # index of current process in the list of llm processes
        self._master_server_rank = llm_master
        self._is_main_server = accelerator.process_index == self._master_server_rank
        self._rl_llm_group = rl_llm_group
        self._rl_llm_group_size = rl_llm_group_size
        self._llm_group = llm_group
        self._llm_group_size = dist.get_world_size(group=self._llm_group)

        # Assign devices
        if config.llm_args.parallelism.use_gpu is False:
            use_cpu = True
            devices = [0]
            lamorel_logger.info("Using CPU on process {} (index {})".format(accelerator.process_index, self._index, devices))
        else:
            use_cpu = False
            devices = self._compute_current_device_map(config)
            lamorel_logger.info("Devices on process {} (index {}): {}".format(accelerator.process_index, self._index, devices))
        self._model = HF_LLM(config.llm_args, devices, use_cpu)
        self._dispatcher = Dispatcher(self._llm_group, self._rl_llm_group_size - 1, self._llm_group_size,
                                      self._is_main_server, self._master_server_rank, self._index)

        custom_module_functions["__score"] = ScoreModuleFunction(self._model.pad_token, config.llm_args.model_type)
        for k, _fn in custom_module_functions.items():
            assert isinstance(_fn, BaseModuleFunction)
            _fn.device = self._model.device
            _fn.llm_config = self._model.get_model_config()
            _fn.initialize()
        self._model.register_module_functions(custom_module_functions)

        if custom_model_initializer is not None:
            self._model = custom_model_initializer.initialize_model(self._model)

        if custom_updater is not None:
            self._updater = custom_updater
            self._updater.set_llm_module(
                DDP(self._model, process_group=self._llm_group,
                    find_unused_parameters=config.allow_subgraph_use_whith_gradient),
            )
            assert isinstance(self._updater, BaseUpdater)
        else:
            self._updater = BaseUpdater()
            self._updater.set_llm_module(self._model)
        self.run()

    def _compute_current_device_map(self, config):
        current_process_index = accelerator.process_index
        # First compute which partition of the local GPUs our current llm process should use
        n_processes = config.distributed_setup_args.n_rl_processes + config.distributed_setup_args.n_llm_processes
        process_ids = np.arange(n_processes)
        machines_processes = np.array_split(process_ids, config.accelerate_args.num_machines)
        current_machine_id = next(i for i, processes in enumerate(machines_processes)
                                  if current_process_index in processes)
        current_machine_processes = list(machines_processes[current_machine_id])
        n_rl_processes = config.distributed_setup_args.n_rl_processes
        if current_machine_processes[0] < n_rl_processes:  # It means we're sharing current node with RL process
            n_shared_rl_processes = len([_p for _p in current_machine_processes if _p < n_rl_processes])
            _local_llm_index = current_machine_processes.index(current_process_index) - n_shared_rl_processes
        else:
            n_shared_rl_processes = 0
            _local_llm_index = current_machine_processes.index(current_process_index)

        # Compute how to partition local GPUs for local LLMs
        cuda_device_ids = np.arange(torch.cuda.device_count())
        processes_devices = np.array_split(cuda_device_ids, len(current_machine_processes) - n_shared_rl_processes)
        current_process_devices = list(processes_devices[_local_llm_index])
        if len(current_process_devices) > config.llm_args.parallelism.model_parallelism_size:
            lamorel_logger.info(
                f"{len(current_process_devices)} gpus available for current LLM but using only model_parallelism_size "
                f"= {config.llm_args.parallelism.model_parallelism_size}")
            current_process_devices = \
                current_process_devices[:config.llm_args.parallelism.model_parallelism_size]

        return current_process_devices

    def _process_calls(self, calls):
        instruction = calls[0]
        if instruction in [InstructionsEnum.FORWARD, InstructionsEnum.GENERATE, InstructionsEnum.UPDATE]:
            calls_data = calls[1]
            if calls_data is None:
                return (instruction, [None])
            else:
                llm_results = []
                for _call in calls_data:
                    if instruction == InstructionsEnum.GENERATE:
                        llm_results.append(self._model.generate(**_call))
                    elif instruction == InstructionsEnum.FORWARD:
                        llm_results.append(self._model(**_call))
                    elif instruction == InstructionsEnum.UPDATE:
                        llm_results.append([self._updater.perform_update(**_call)])
                return (instruction, llm_results)
        elif instruction == InstructionsEnum.CLOSE:
            lamorel_logger.info("Closing LLM server process {}".format(accelerator.process_index))
            sys.exit()
        else:
            raise NotImplementedError('Unknown provided instruction.')

    def run(self):
        lamorel_logger.info("Launching LLM server process {}".format(accelerator.process_index))
        while True:
            #### Receive calls from RL processes and dispatch them over LLMs ####
            method_calls = [None for _ in range(self._rl_llm_group_size)]
            if self._is_main_server:
                dist.gather_object(
                    obj=None, object_gather_list=method_calls, dst=accelerator.process_index, group=self._rl_llm_group
                )
                method_calls = method_calls[:-1]  # remove last one coming from current process
                assert len(set([call["instruction"] for call in method_calls])) <= 1  # check all calls are the same
            calls_to_process = self._dispatcher.dispatch(method_calls)
            current_process_results = self._process_calls(calls_to_process)
            if current_process_results[1] is not None:  # expected answer from caller
                gathered_results = self._dispatcher.gather(current_process_results)
                if self._is_main_server:
                    assert len(gathered_results) == self._rl_llm_group_size-1
                    if method_calls[0]["instruction"] in [InstructionsEnum.FORWARD, InstructionsEnum.GENERATE]:
                        for idx, _call in enumerate(method_calls):
                            if 'candidates' in _call:
                                if "__score" in method_calls[0]["module_function_keys"]:
                                    for i in range(len(_call["contexts"])):
                                        assert len(gathered_results[idx][i]["__score"]) == len(_call["candidates"][i])
                            else: # enough generations
                                assert len(_call["contexts"]) == len(gathered_results[idx])

                    dist.broadcast_object_list(object_list=gathered_results + [None], src=self._master_server_rank,
                                               group=self._rl_llm_group)


