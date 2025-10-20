import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch
import sys
from omegaconf import OmegaConf
import copy

import logging
lamorel_logger = logging.getLogger('lamorel_logger')

from .llms.handlers import HandlersEnum
from .llms.updaters import BaseUpdater
from .llms.module_functions import BaseModuleFunction, ScoreModuleFunction
from .dispatcher import Dispatcher
from .utils import InstructionsEnum

class Server:
    def __init__(self, rank, name, llm_config, distributed_config, llm_index, llm_group, llm_master_rank, rl_llm_group, rl_llm_group_size, custom_updater,
                 custom_module_functions, custom_model_initializer):
        assert dist.is_initialized(), "torch distributed must be used!"
        self.rank = rank
        self.name = name
        self._index = llm_index  # index of current process in the list of llm processes
        self._master_server_rank = llm_master_rank
        self._is_main_server = self.rank == self._master_server_rank
        self._rl_llm_group = rl_llm_group
        self._rl_llm_group_size = rl_llm_group_size
        self._llm_group = llm_group
        self._llm_group_size = dist.get_world_size(group=self._llm_group)

        # Assign devices
        devices = distributed_config.devices_per_process[self._index]
        if devices == "cpu":
            use_cpu = True
            lamorel_logger.info("Using CPU on  LLM '{}' index {} (rank {})".format(self.name, self._index, self.rank))
        else:
            assert all([isinstance(_device, int) for _device in devices])
            use_cpu = False
            lamorel_logger.info("Devices on process LLM '{}' index {} (rank {}): {}".format(self.name, self._index, self.rank, devices))

        model_class = HandlersEnum[llm_config.handler].value
        self._model = model_class(llm_config, devices, self.rank, use_cpu)
        self._dispatcher = Dispatcher(self._llm_group, self._rl_llm_group_size, self._llm_group_size,
                                      self._is_main_server, self._master_server_rank, self._index)

        current_process_config = {
            "llm": name,
            "process_index": rank
        }
        _llm_copy_to_pass = copy.copy(llm_config)
        for __k, __v in self._model.get_additional_llm_config().items():
            OmegaConf.update(_llm_copy_to_pass, __k, __v, force_add=True)
        custom_module_functions["__score"] = ScoreModuleFunction(self._model.pad_token)
        for k, _fn in custom_module_functions.items():
            assert isinstance(_fn, BaseModuleFunction)
            _fn.device = self._model.main_device
            _fn.llm_config = _llm_copy_to_pass
            _fn.current_process_config = current_process_config
            _fn.model_config = self._model.get_model_config()
            _fn.initialize()
        self._model.register_module_functions(custom_module_functions)

        if custom_model_initializer is not None:
            custom_model_initializer.llm_config = _llm_copy_to_pass
            custom_model_initializer.current_process_config = current_process_config
            custom_model_initializer.model_config = self._model.get_model_config()
            self._model = custom_model_initializer.initialize_model(self._model)

        if custom_updater is not None:
            self._updater = custom_updater
            self._updater.llm_config = _llm_copy_to_pass
            self._updater.current_process_config = current_process_config
            self._updater.model_config = self._model.get_model_config()
            if dist.get_backend(group=self._llm_group) == "gloo":
                lamorel_logger.info(f"Ignoring boolean buffers for DDP on LLM {self.name} (index {self._index}) as GLOO backend is used.")
                self._model._ddp_params_and_buffers_to_ignore = [name for name, buffer in self._model.named_buffers() if
                                                                 buffer.dtype == torch.bool]  # This is the trick, you ask DDP to ignore all buffers that are in torch.bool because GLOO doesn't support bool.

            ddp_kwargs = {}
            if distributed_config.ddp_kwargs is not None:
                ddp_kwargs.update(distributed_config.ddp_kwargs)
            self._updater.set_llm_module(
                DDP(self._model, process_group=self._llm_group, **ddp_kwargs)
            )
            assert isinstance(self._updater, BaseUpdater)
        else:
            self._updater = BaseUpdater()
            self._updater.set_llm_module(self._model)
        self.run()

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
            lamorel_logger.info("Closing server for LLM '{}' index {} (rank {})".format(self.name, self._index, self.rank))
            sys.exit()
        else:
            raise NotImplementedError('Unknown provided instruction.')

    def run(self):
        if self._is_main_server:
            lamorel_logger.info(
                "Launching master server for LLM '{}' index {} (rank {})".format(self.name, self._index, self.rank))
        else:
            lamorel_logger.info(
                "Launching slave server for LLM '{}' index {} (rank {})".format(self.name, self._index, self.rank))
        while True:
            #### Receive calls from RL processes and dispatch them over LLMs ####
            method_calls = [None for _ in range(self._rl_llm_group_size + 1)]
            if self._is_main_server:
                dist.gather_object(
                    obj=None, object_gather_list=method_calls, dst=self.rank, group=self._rl_llm_group
                )
                method_calls = method_calls[:-1]  # remove last one coming from current process
                assert len(set([call["instruction"] for call in method_calls])) <= 1  # check all calls are the same
            calls_to_process = self._dispatcher.dispatch(method_calls)
            current_process_results = self._process_calls(calls_to_process)
            if current_process_results[1] is not None:  # expected answer from caller
                gathered_results = self._dispatcher.gather(current_process_results)
                if self._is_main_server:
                    assert len(gathered_results) == self._rl_llm_group_size
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


