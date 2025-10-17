import torch.distributed as dist
import typing
from .server import Server
from .server.utils import InstructionsEnum
from lamorel.utils.init_distributed_setup import init_distributed_setup
import os

import logging
lamorel_logger = logging.getLogger('lamorel_logger')

class Caller:
    '''
    This class should be called by each process.
    It will instantiate the different distributed groups.
    If the current process belongs to the LLM's processes, it will launch the LLM and wait for requests.
    '''
    def __init__(self, config, custom_updater={}, custom_module_functions={}, custom_model_initializer={}):
        if not dist.is_initialized():
            backend, timeout = init_distributed_setup(config.distributed_setup_args)
        else:
            raise Exception("The distributed process group is already initialized. This is likely due to a second instantiation of the caller.")

        self.__config = config
        self.current_process_llm_group_name = None
        self.rank = int(os.environ["RANK"])
        # self.local_rank = int(os.environ["LOCAL_RANK"])

        # Set log level
        numeric_log_level = getattr(logging, config.log_level.upper(), None)
        if not isinstance(numeric_log_level, int):
            raise ValueError('Invalid log level: %s' % config.log_level)
        lamorel_logger.setLevel(numeric_log_level)

        # Initialize distributed groups
        # RL processes are considered as the first n processes
        rl_processes = list(range(config.distributed_setup_args.n_rl_processes))

        lamorel_logger.info("Init rl group for process {}".format(self.rank))
        self._rl_group = dist.new_group(
            ranks=rl_processes,
            backend=backend,
            timeout=timeout
        )

        self._llm_groups = {}
        offset = config.distributed_setup_args.n_rl_processes
        for _llm_name, _llm_config in config.distributed_setup_args.llm_processes.items():
            ranks = list(range(offset, offset + _llm_config.n_processes))
            offset = ranks[-1] + 1
            lamorel_logger.info("Init llm group {} for process {}".format(_llm_name, self.rank))
            lamorel_logger.info("Init rl-llm group {} for process {}".format(_llm_name, self.rank))
            self._llm_groups[_llm_name] = {
                "ranks": ranks,
                "master_rank": ranks[0],
                "llm_group": dist.new_group(
                    ranks=ranks,
                    backend=backend,
                    timeout=timeout
                ),
                "rl_llm_group": dist.new_group(
                    ranks=rl_processes + [ranks[0]],
                    backend=backend,
                    timeout=timeout
                )
            }

            if self.rank in ranks:
                self.current_process_llm_group_name = _llm_name


        if self.current_process_llm_group_name is not None:
            current_group = self._llm_groups[self.current_process_llm_group_name]
            Server(
                self.rank,
                self.current_process_llm_group_name,
                config.llm_configs[self.current_process_llm_group_name],
                config.distributed_setup_args.llm_processes[self.current_process_llm_group_name],
                current_group["ranks"].index(self.rank),
                current_group["llm_group"],
                current_group["master_rank"],
                current_group["rl_llm_group"],
                len(rl_processes),
                custom_updater[self.current_process_llm_group_name] if self.current_process_llm_group_name in custom_updater else None,
                custom_module_functions[self.current_process_llm_group_name] if self.current_process_llm_group_name in custom_module_functions else {},
                custom_model_initializer[self.current_process_llm_group_name] if self.current_process_llm_group_name in custom_model_initializer else None
            )

    def get_rl_dist_group(self):
        '''
        Use this group for communication between RL processes.
        :return: torch distributed group
        '''
        return self._rl_group

    def score(self, contexts: typing.List[str], candidates: typing.List[typing.List[str]] = None,
              pad_contexts: bool = True, forced_context_len: int = None, forced_sequence_len: int = None,
              add_eos_on_candidates: bool = False, require_grad: bool = False, minibatch_size: int = None,
              additional_module_function_keys: typing.List[str] = [], peft_adapter: str = None,
              llm_to_call : str = None, **kwargs):
        '''
        Returns probabilities or log probabilities for each candidate to follow its context.
        Params:
        - `contexts`: Input sentences.
        - `candidates` (default=None): Associated output sentences.
        - `pad_contexts` (default=True): Whether contexts should be padded (left for causal, right for seq2seq) to all have the same length. If set to `True`, contexts will be padded to have the same size (either `forced_context_len` if not `None` or the longest context's size). This parameter is ignored if `model_type="seq2seq"` or `pre_encode_inputs=True`.
        - `forced_context_len` (default=None): Length up to which contexts must be padded. Used if `pad_contexts=True` or `model_type="seq2seq"` or `pre_encode_inputs=True`.
        - `forced_sequence_len` (default=None): With decoder-only models, when candidates are appended to contexts, they are right padded so that complete sequences all have the same size. You can use this parameter to force the size of the complete sequence. Otherwise, it will be padded to match the size of the longest sequence in the batch.
        - `add_eos_on_candidates` (default=False): Whether the EOS token should be appended to candidates or not.
        - `require_grad` (default=False): `require_grad` value to set on the LLM's (and CustomModuleFunctions) parameters.
        - `minibatch_size` (default=None): Custom minibatch_size to use (if None, the one set in lamorel's config will be used).
        - `additional_module_function_keys` (default=[]): Additional CustomModuleFunctions to compute. If any is passed, a dictionary with one key per CutomModuleFunction will be returned for each context (with the key `__score` containing the scores).
        - `peft_adapter` (default=None): If not None, the Peft Adapters' name to activate.
        - `llm_to_call` (default=None): The name of the LLM (specified in lamorel's config) to use. None will be accepted if only a single LLM is used.
        '''
        module_function_keys = ["__score"]
        module_function_keys.extend(additional_module_function_keys)
        result = self.__call_model(InstructionsEnum.FORWARD, True,
                                   module_function_keys=module_function_keys,
                                   contexts=contexts,
                                   candidates=candidates,
                                   add_eos_on_candidates=add_eos_on_candidates,
                                   require_grad=require_grad,
                                   minibatch_size=minibatch_size,
                                   peft_adapter=peft_adapter,
                                   pad_contexts=pad_contexts,
                                   llm_to_call=llm_to_call,
                                   **kwargs)
        if additional_module_function_keys == []:
            result = [_r['__score'] for _r in result]
        return result

    def generate(self, contexts: list, return_logprobs: bool = False, peft_adapter: str = None, llm_to_call : str = None,
                 **kwargs):
        '''
        Returns for each context a list of dict containing for each generated sequence:
        - `tokens`: the sampled tokens
        - `text`: the generated text as a string (once the LLM's tokenizer has been used on `tokens`)

        If `return_logprobs=False` is passed, this dict contains two additional keys:
        - `tokens_probability`: the probability of each token in the sequence
        - `text_probability`: the probability of the whole sequence (i.e. the product of `tokens_probability`)

        Otherwise, the dict contains the following two additional keys:
        - `tokens_logprob: the log probability of each token in the sequence
        - `text_logprob: the log probability of the whole sequence (i.e. the sum of `tokens_logprob`)

        Params:
        - `contexts`: Input sentences.
        - `return_logprobs` (default=False): Whether normalized probabilities or log probabilities should be returned.
        - `peft_adapter` (default=None): If not None, the Peft Adapters' name to activate.
        - `llm_to_call` (default=None): The name of the LLM (specified in lamorel's config) to use. None will be accepted if only a single LLM is used.
        '''
        return self.__call_model(InstructionsEnum.GENERATE, True, contexts=contexts,
                                 return_logprobs=return_logprobs, peft_adapter=peft_adapter, llm_to_call=llm_to_call,
                                 **kwargs)

    def update(self, contexts: typing.List[str], candidates: typing.List[typing.List[str]] = None, llm_to_call : str = None,
               **kwargs):
        '''
        Triggers the Updater if one has been passed.
        Params:
        - `contexts`: Input sentences.
        - `candidates`: Associated output sentences.
        - `llm_to_call` (default=None): The name of the LLM (specified in lamorel's config) to use. None will be accepted if only a single LLM is used.
        '''
        result = self.__call_model(InstructionsEnum.UPDATE, True, contexts=contexts, candidates=candidates,
                                   llm_to_call=llm_to_call, **kwargs)
        if not isinstance(result, list):
            result = [result]
        return result

    def close(self, llm_to_call : str = None):
        '''
        Stop an LLM's processes.
        Params:
        - `llm_to_call` (default=None): The name of the LLM (specified in lamorel's config) to use. If None, all the LLMs will be stopped.
        '''
        if llm_to_call is None:
            for _llm_name in self._llm_groups.keys():
                self.__call_model(InstructionsEnum.CLOSE, llm_to_call=_llm_name, expect_answer=False)
        else:
            self.__call_model(InstructionsEnum.CLOSE, llm_to_call=llm_to_call, expect_answer=False)

    def custom_module_fns(self, module_function_keys : typing.List[str], contexts: typing.List[str],
                          candidates: typing.List[typing.List[str]] = None, pad_contexts: bool = True,
                          forced_context_len: int = None, forced_sequence_len: int = None,
                          add_eos_on_candidates: bool = False, require_grad: bool = False, minibatch_size: int = None,
                          peft_adapter: str = None, llm_to_call : str = None, **kwargs):
        '''
        Triggers the forward pass of an LLM along with one or multiple CustomModuleFunctions for several input sentences (contexts).
        Params:
        - `module_function_keys`: CustomModuleFunctions' name.
        - `contexts`: Input sentences.
        - `candidates` (default=None): Associated output sentences.
        - `pad_contexts` (default=True): Whether contexts should be padded (left for causal, right for seq2seq) to all have the same length. If set to `True`, contexts will be padded to have the same size (either `forced_context_len` if not `None` or the longest context's size). This parameter is ignored if `model_type="seq2seq"` or `pre_encode_inputs=True`.
        - `forced_context_len` (default=None): Length up to which contexts must be padded. Used if `pad_contexts=True` or `model_type="seq2seq"` or `pre_encode_inputs=True`.
        - `forced_sequence_len` (default=None): With decoder-only models, when candidates are appended to contexts, they are right padded so that complete sequences all have the same size. You can use this parameter to force the size of the complete sequence. Otherwise, it will be padded to match the size of the longest sequence in the batch.
        - `add_eos_on_candidates` (default=False): Whether the EOS token should be appended to candidates or not.
        - `require_grad` (default=False): `require_grad` value to set on the LLM's (and CustomModuleFunctions) parameters.
        - `minibatch_size` (default=None): Custom minibatch_size to use (if None, the one set in lamorel's config will be used).
        - `peft_adapter` (default=None): If not None, the Peft Adapters' name to activate.
        - `llm_to_call` (default=None): The name of the LLM (specified in lamorel's config) to use. None will be accepted if only a single LLM is used.
        '''
        return self.__call_model(InstructionsEnum.FORWARD, True,
                                 module_function_keys=module_function_keys,
                                 contexts=contexts,
                                 candidates=candidates,
                                 add_eos_on_candidates=add_eos_on_candidates,
                                 require_grad=require_grad,
                                 minibatch_size=minibatch_size,
                                 peft_adapter=peft_adapter,
                                 pad_contexts=pad_contexts,
                                 llm_to_call=llm_to_call,
                                 **kwargs)

    def __call_model(self, instruction, expect_answer, llm_to_call=None, **kwargs):
        if llm_to_call is None:
            if len(self._llm_groups.keys()) == 1:
                llm_to_call = list(self._llm_groups.keys())[0]
            else:
                raise Exception("Multiple LLMs exist, please specify which one to call")

        llm_group_to_call = self._llm_groups[llm_to_call]
        dist.gather_object(
            obj={
                "instruction": instruction,
                **kwargs
            }, object_gather_list=None, dst=llm_group_to_call["master_rank"], group=llm_group_to_call["rl_llm_group"]
        )

        if expect_answer:
            results = [None for _ in range(dist.get_world_size(group=llm_group_to_call["rl_llm_group"]))]
            dist.broadcast_object_list(object_list=results,
                                       src=llm_group_to_call["master_rank"],
                                       group=llm_group_to_call["rl_llm_group"])
            return results[self.rank]
        else:
            return None

