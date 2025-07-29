import torch.distributed as dist
from copy import copy, deepcopy
from enum import Enum
import numpy as np
import torch
from .utils import InstructionsEnum

class DispatchMode(Enum):
    calls = 0,
    batchs = 1


class Dispatcher:
    def __init__(self, process_group, n_callers, n_handlers, is_current_process_master, master_rank,
                 current_process_index):
        self._process_group = process_group
        self._n_callers = n_callers
        self._n_handlers = n_handlers
        self._is_master = is_current_process_master
        self._master_rank = master_rank
        self._current_process_index = current_process_index

        if self._n_callers >= self._n_handlers:
            self._mode = DispatchMode.calls
            self._n_calls_per_handler = self._n_callers // self._n_handlers
        else:
            self._mode = DispatchMode.batchs
            self._current_calls_infos = []
            self._n_handlers_per_call = self._n_handlers // self._n_callers

    def __dispatch_calls(self, calls):
        '''
        Dispatch `n_callers` calls over `n_handlers` llms
        '''
        if self._is_master:
            scattered_calls = []
            for i in range(0, len(calls), self._n_calls_per_handler):
                _current_handler_calls = []
                for j in range(self._n_calls_per_handler):
                    _call = {
                        k: v for k, v in calls[i+j].items()
                        if k not in ["instruction"]
                    }
                    if calls[i+j]["instruction"] in [InstructionsEnum.UPDATE]:
                        _call["_current_batch_ids"] = [i for i in range(len(_call["contexts"]))]
                    _current_handler_calls.append(_call)
                scattered_calls.append(_current_handler_calls)
        else:
            scattered_calls = [None for _ in range(self._n_handlers)]
        return scattered_calls

    def __gather_calls(self, calls):
        '''
        Gather `n_callers` results from `n_handlers` llms
        '''
        gathered_calls = []
        if self._is_master:
            for _calls in calls:
                gathered_calls.extend(_calls)
        return gathered_calls

    def __split_generations(self, n, m, k):
        '''
        n: mumber of contexts
        m: num_return_seqences
        k: number of handlers

        split generations over Handlers, when k > n for parallelised generate
        '''
        assert k >= n, "condition k > n not respected"

        handlers_per_context = [1] * n

        leftover = k - n
        idx = 0

        while leftover > 0:
            handlers_per_context[idx % n] += 1
            leftover -= 1
            idx += 1

        tasks = []
        handler_id = 0
    
        for context_id, num_handlers  in enumerate(handlers_per_context):
            base = m // num_handlers
            remainder = m % num_handlers
            left_generations = m

            for i in range(num_handlers):
                if left_generations == 0: break
                num_generations = base + (1 if i < remainder else 0)
                tasks.append({
                    "handler_id": handler_id,
                    "context_id": context_id,
                    "num_return_sequences": num_generations
                })
                left_generations -= num_generations
                handler_id += 1

        return tasks




    def __dispatch_batches(self, calls):
        '''
        When `n_callers` < `n_handlers` we can dispatch each call over multiple LLMs.
        We first check for each call whether multiple entries are sent (each caller can send a batch of entries) and dispatch them.
        If we still have enough LLMs, we check if the `score` method is called (i.e. with multiple candidates). If so,
        we dispatch the candidates over multiple LLMs to score minibatches in parallel.

        When n_handlers > n_callers we dispatch llms over calls, using __split_generation function, 
        We firstly assign ( n_handlers // n_calls handler) to each call, then we asign the rest to the first calls one handler each.
        for calls having more thean one handler we dispatch return_sequences over handlers
        '''
        scattered_calls = [None for _ in range(self._n_handlers)]
        if self._is_master:
            i = 0
            for call in calls:
                _batch_size = len(call["contexts"])
                self._current_calls_infos.append({  # Useful for for gathering
                    'batch_size': _batch_size
                })
                dispatched_call = {
                    k: v for k, v in call.items()
                    if k not in ["instruction", "contexts", "candidates"]
                }
                if self._n_handlers_per_call / _batch_size > 1:  # Each batch entry should be splitted over multiple handlers
                    _ids = np.arange(i, i + self._n_handlers_per_call)
                    batches_handlers = np.array_split(_ids, _batch_size)
                    for j in range(len(call["contexts"])):
                        _dispatched_call = copy(dispatched_call)
                        _dispatched_call["contexts"] = [call["contexts"][j]]
                        _batch_handlers = batches_handlers[j]
                        if "candidates" in call:
                            _call_batch_ids = np.arange(len(call["candidates"][j]))
                            _minibatches = np.array_split(_call_batch_ids, len(_batch_handlers))
                            for h_idx, _handler in enumerate(_batch_handlers):
                                _call = deepcopy(_dispatched_call)
                                _call["candidates"] = [[call["candidates"][j][_idx] for _idx in _minibatches[h_idx]]]
                                scattered_calls[_handler] = [_call]
                         
                        #Parallelised generate when n_llms > len(contexts)"
                        elif call["instruction"] == InstructionsEnum.GENERATE:
                                n_seq = call["num_return_sequences"]
                                num_contexts = len(call['contexts'])
                                splits = self.__split_generations(num_contexts, n_seq, self._n_handlers)
                                self._current_calls_infos[j]["generation_map"] = splits
                                for task_idx, task in enumerate(splits):
                                    _dispatched_call = copy(dispatched_call)
                                    _dispatched_call["num_return_sequences"] = task["num_return_sequences"]
                                    _dispatched_call["contexts"] = [ call["contexts"][task["context_id"]] ]  
                                    scattered_calls[i + task_idx] = [_dispatched_call]
                        else:    
                            raise NotImplementedError("Case Not Implemented")
                            
                else:  # Each handler handles multiple batch entry
                    _ids = np.arange(_batch_size)
                    batch_chunks = np.array_split(_ids, self._n_handlers_per_call)
                    for j in range(i, i + self._n_handlers_per_call):
                        _dispatched_call = copy(dispatched_call)
                        _dispatched_call["contexts"] = [call["contexts"][_idx] for _idx in batch_chunks[j]]
                        if "candidates" in call:
                            if call["candidates"] is not None:
                                _dispatched_call["candidates"] = [call["candidates"][_idx] for _idx in batch_chunks[j]]
                            else:
                                _dispatched_call["candidates"] = None
                        if call["instruction"] in [InstructionsEnum.UPDATE]:
                            _dispatched_call["_current_batch_ids"] = batch_chunks[j].tolist()
                        scattered_calls[j] = [_dispatched_call]

                i += self._n_handlers_per_call
        return scattered_calls


    def __gather_batches(self, calls):
        '''' Gather responses after dispatching calls'''
        gathered_calls = []
        if self._is_master:

            step = self._n_handlers_per_call           

            for i in range(0, self._n_handlers, step):

                meta = self._current_calls_infos[i]      
                instr = meta["instruction"]

                gen_map = meta.get("generation_map")
                # gather calls based on _split_generation dispatching
                if instr == InstructionsEnum.GENERATE and gen_map:

                    n_ctx   = meta["batch_size"]
                    ctx_out = [[] for _ in range(n_ctx)]

                    for task in gen_map:
                        h_global = i + task["handler_id"]   
                        part     = calls[h_global][0]          
                        
                        if part is None:
                            continue
                            
                        if isinstance(part,list):
                            if len(part) == 1 and isinstance(part[0], list):
                                seqs = part[0]
                            else:
                                seqs = part
                        elif isinstance(part,dict) and "__generate" in part:
                            seqs = part["__generate"]
                        else:
                            raise ValueError(f"Unsupported type for GENERATE : {type(part)}")

                        ctx_out[task["context_id"]].extend(seqs)
                    gathered_calls.append(ctx_out)

                    continue
                

                # (FORWARD / UPDATE / score)
                current_call_results = []
                batch_size = meta["batch_size"]

                if step / batch_size > 1:
                    ids          = np.arange(i, i + step)
                    batches_split = np.array_split(ids, batch_size)
                    for handlers_for_ctx in batches_split:
                        ctx_dict = {}
                        for h in handlers_for_ctx:
                            part = calls[h][0]
                            if part is not None and isinstance(part[0], dict):
                                for k, v in part[0].items():
                                    ctx_dict[k] = torch.cat((ctx_dict[k], v), 0) if k in ctx_dict else v
                        current_call_results.append(ctx_dict)
                else:
                    for h in range(i, i + step):
                        current_call_results.extend(calls[h][0])

                gathered_calls.append(current_call_results)

        # reset meta data
        self._current_calls_infos = []
        return gathered_calls


    def dispatch(self, calls):
        instruction = None
        if self._is_master:
            instruction = calls[0]["instruction"]

        if instruction in [InstructionsEnum.FORWARD, InstructionsEnum.GENERATE, InstructionsEnum.UPDATE]:
            if self._mode == DispatchMode.calls:
                _scattered_calls = self.__dispatch_calls(calls)
            else:
                _scattered_calls = self.__dispatch_batches(calls)
            scattered_calls = [(instruction, _call) for _call in _scattered_calls]
        else:
            scattered_calls = [(instruction,) for _ in range(self._n_handlers)]
        #### Scatter over processes ####
        # TODO Open an issue on torch distributed for `scatter_object_list` not working
        # calls_to_process = [{} for _ in range(n_calls_per_llm)]
        # dist.scatter_object_list(scatter_object_output_list=calls_to_process,
        #                          scatter_object_input_list=scattered_calls,
        #                          src=self._master_server_rank, group=self._llm_group)
        # Get calls to be processed from current process
        dist.broadcast_object_list(scattered_calls, src=self._master_rank, group=self._process_group)
        return scattered_calls[self._current_process_index]

    def gather(self, handler_results):
        instruction = handler_results[0]

        if self._is_master:
            handlers_results = [None for _ in range(self._n_handlers)]
        else:
            handlers_results = None
        dist.gather_object(handler_results, handlers_results, dst=self._master_rank, group=self._process_group)

        if self._is_master:
            results = [_handler_results[1] for _handler_results in handlers_results]
            if instruction in [InstructionsEnum.FORWARD, InstructionsEnum.GENERATE, InstructionsEnum.UPDATE]:
                if self._mode == DispatchMode.calls:
                    return self.__gather_calls(results)
                else:
                    call_results = self.__gather_batches(results)
                    if instruction == InstructionsEnum.GENERATE:
                        call_results = list(map(
                            lambda call_batch: [_results['__generate'] for _results in call_batch],
                            call_results
                        ))
                    return call_results
            else:
                raise NotImplementedError()
        else:
            return None