import os
import torch
from torch.nn.functional import log_softmax

import logging
lamorel_logger = logging.getLogger('lamorel_logger')

#from .utils import load_hf_model_and_tokenizer
from base_llm import BaseLLM

import vllm
from vllm import SamplingParams
from math import exp

import torch
from contextlib import nullcontext


from transformers import set_seed

# Accelerate
from accelerate import infer_auto_device_map, init_empty_weights, Accelerator
from contextlib import nullcontext


from torch.nn.functional import log_softmax




class VLLM(BaseLLM):
    """
    This class is a wrapper around the language model using VLLM inplementation.
    """
    def __init__(self, args, devices, use_cpu):
        if use_cpu:
            raise NotImplementedError("No CPU pre-built image availaible with VLLM, try with Hugging face Transformers !")
        
        super().__init__(args, devices, use_cpu)

        seed = 42 if args.seed is None else args.seed
        set_seed(seed)

        print("Parallelising HF LLM on {} devices".format(len(self.devices)))
        print(f"Loading model with dtype {args.dtype}")        


        if args.model_type not in ["causal"]:
            raise NotImplementedError("VLLM only support  causal models, try with Hugging face Transformers !")
        self.model_type = "causal"

        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, devices))
        self._LLM_model = vllm.LLM(
            model= args.model_path,
            dtype= args.dtype,  
            tensor_parallel_size= args.parallelism.model_parallelism_size
        )
        
        self._LLM_tokenizer =  self._LLM_model.get_tokenizer()
        self.__LLM_config = self._LLM_model.llm_engine.model_config 
        
        
        self._scoring_minibatch_size = args.minibatch_size
        if args.model_type == "causal":
            if args.pre_encode_inputs:
                lamorel_logger.info("Pre_encode_inputs not availaible on vLLM backend, this option will be avoided")
                
        else: 
            raise NotImplementedError("VLLM only supports Causal Models")
        
        if self._LLM_tokenizer.pad_token is not None:
            self.pad_token = self._LLM_tokenizer.pad_token_id
        else:
            self.pad_token = self._LLM_tokenizer(" ", add_special_tokens=False)["input_ids"][0]

        self.__synchronize_gpus_after_scoring = args.parallelism.synchronize_gpus_after_scoring
        self.__empty_cuda_cache_after_scoring = args.parallelism.empty_cuda_cache_after_scoring
        
        # Handling seeds for sampling 
        lamorel_logger.info("Setting torch seed to seed + process_index")
        set_seed(seed + self.accelerator.process_index)


    def get_model_config(self):
        """
        returns the config of the loaded LLM, as a ModelConfig Object
        """
        return self.__LLM_config

    def register_module_functions(self, module_functions):
        """
        this function sets he module funciton to apply on scores.
        NB: module_functions is a  dict(<key,module>) where module is a subClass of torch.nn.Module.
        """
        self._module_functions = torch.nn.ModuleDict(module_functions)

    def get_trainable_module(self):
        """
        returns the model instance. aka vllm.LLM object
        """
        return self._LLM_model

    def generate(self, contexts, return_logprobs: bool = False,**kwargs):
        """
        Implements lmorel_server.generate with a vllm backend, 
        the kwargs are must be coherent with Sampling_params of the Vllm API.
        except num_return_sequences as it's used in the dispatcher.  
        """
        
        sp_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        sp_kwargs["logprobs"] = 1 if kwargs.get("logprobs") is None else kwargs["logprobs"]        
        sp_kwargs["prompt_logprobs"] = 0
        
        "The parameter num_return_sequences was used in the dispatcher, now must be converted to a Sampling_params attribute"
        if sp_kwargs.get("num_return_sequences"):
            sp_kwargs["n"] = sp_kwargs["num_return_sequences"]
            sp_kwargs.pop("num_return_sequences")

        sampling_params = SamplingParams(**sp_kwargs)

        vllm_outputs = self._LLM_model.generate(contexts, sampling_params)
        
        generations = [[] for _ in range(len(contexts))]

        for req in vllm_outputs:                           
            prompt_idx = int(req.request_id)                    

            for comp in req.outputs:                       
                text   = comp.text
                tokens = comp.token_ids
                tok_logprobs = None
                
                if comp.logprobs is not None:
                    tok_logprobs = []
                    for i, t_id in enumerate(tokens):
                        tok_logprobs.append(comp.logprobs[i][t_id].logprob)
                
                res = {
                    "text": text,
                    "tokens": tokens,
                    "text_logprob": comp.cumulative_logprob
                                    if comp.cumulative_logprob is not None
                                    else sum(tok_logprobs) if tok_logprobs else None,
                    "tokens_logprob": tok_logprobs, 
                }

                if not return_logprobs:
                    res = {
                        "text": res["text"],
                        "tokens": res["tokens"],
                        "text_probability": exp(res["text_logprob"]),
                        "tokens_probability": [exp(x) for x in res["tok_logprobs"]],                        
                    }

                # Beam score 
                if hasattr(comp, "score") and comp.score is not None:
                    res["beam_score"] = comp.score

                generations[prompt_idx].append(res)
        
        lamorel_logger.debug(f"Generating finished on process {self.accelerator.process_index}")

        return generations

    def score(self,  contexts, candidates, module_function_keys=[], require_grad=False, minibatch_size=None, **kwargs):
            """
            Implements the lamorel's score using vllm backend.
            The score function relies on the vllm.generate method with prompt = context + condidate 
            Compute log p(condidate|context) via prompt_logprobs as log p(candidate + context) -  p (context)
            """

            _forward_results = [{} for _ in range(len(contexts))]

            if candidates is None or len(candidates) == 0:
                candidates = [[""] for _ in contexts]
            else:
                assert len(candidates) == len(contexts), \
                    "If candidates are provided, there should be one list of candidates per context."
                if any(len(_c) == 0 for _c in candidates):
                    candidates = [[""] if len(_c) == 0 else _c for _c in candidates]

            _ids_tables = {}
            offset = 0
            for i, _cands in enumerate(candidates):
                _ids_tables[i] = list(range(offset, offset + len(_cands)))
                offset += len(_cands)

            flat_prompts_full = []
            flat_prompts_ctx = []

            """
            If a context ends with a trailing space ' ', this token will be present in prompt_token_ids.
            However, when we concatenate the context with a candidate, the trailing space token may not
            be preserved in the tokenization, leading to inconsistent logprob results due to tokenization
            differences between the context alone and the context-candidate combination.
            """
            gap_space = False
            for ctx, cand_list in zip(contexts, candidates):
                if ctx[-1] == " ":
                    gap_space = True
                    ctx = ctx[:-1]
                for cand in cand_list:
                    flat_prompts_full.append(ctx + cand)
                flat_prompts_ctx.extend([ctx] * len(cand_list))
            if gap_space == True :
                lamorel_logger.info("Score: In order to avoid an inconsistent scores, the trailing space ' ' at the end of your contexts are removed ")

            def _sum_prompt_logprobs(vllm_outputs):
                """compute the cumulative logprobs of the prompt tokens ignoring BOS."""
                sums = []
                for out in vllm_outputs:
                    token_ids = getattr(out, "prompt_token_ids", None)
                    plps = getattr(out, "prompt_logprobs", None)
                    if token_ids is None or plps is None:
                        sums.append(float("nan"))
                        continue
                    start = 1 if len(token_ids) > 0 else 0  # ignore BOS 
                    cur = 0.0
                    for i in range(start, len(token_ids)):
                        lp_dict = plps[i]
                        tid = token_ids[i]
                        cur += lp_dict.get(tid).logprob if (lp_dict and tid in lp_dict) else float("-inf")
                    sums.append(cur)
                return torch.tensor(sums, dtype=torch.float32)

            _forward_results = [] * len(contexts)
            total_size = len(flat_prompts_full)
            minibatch_size = self._scoring_minibatch_size
            m = ( total_size + minibatch_size - 1) // minibatch_size #adding minibatch_size -1 only to avoid start=end

            with torch.no_grad() if not require_grad else nullcontext():
                sp = { k:v for k,v in kwargs if v is not None }
                #these two parameters are necessary to get the log_probs
                sp["prompt_logprobs"] = 0
                sp["logprobs"] = 1

                out_full = []

                # 1) log p(context + candidate)
                sp = SamplingParams(**sp)
                for m_batch_idx in range(m ):
                    start_idx = m_batch_idx * minibatch_size
                    end_idx = min(start_idx + minibatch_size, total_size)
                    mb_out_full = self._LLM_model.generate(flat_prompts_full[start_idx: end_idx], sp)
                    out_full.extend(mb_out_full)
                
                lp_full = _sum_prompt_logprobs(out_full)
                
                offset = 0
                ctx_sum = []

                ## 2) log p(context)
                # We compute the context's log_probs using the ouput of context + candidate, instead of forwarding the context
                for idx, ctx in enumerate(contexts):
                    step = len(candidates[idx])
                    ctx_tokens = self._LLM_model.llm_engine.tokenizer.encode(ctx,add_special_tokens=False)
                    first_output_idx = offset
                    output = out_full[first_output_idx]
                    plps = getattr(output, "prompt_logprobs", None)
                    start = 1 if len(ctx_tokens) > 0 else 0
                    cur = 0.0
                    for i in range(start, len(ctx_tokens)):
                        lp_dict = plps[i]
                        tid = ctx_tokens[i-start]
                        cur += lp_dict.get(tid).logprob if (lp_dict and tid in lp_dict) else 0
                    ctx_sum.extend([cur] * step )
                    offset += step

                lp_ctx = torch.tensor(ctx_sum, dtype=torch.float32)

                # 3) log p(candidate | context) = log p(candidae + context) - log p(context)
                cond_ll = lp_full - lp_ctx  
                
            for k,v in _ids_tables.items():    
                _forward_results.append(cond_ll[v])
            
            if self.__synchronize_gpus_after_scoring:
                lamorel_logger.debug(f"Synchronizing GPUs on process {self.accelerator.process_index}")
                for device in self.devices:
                    torch.cuda.synchronize(device)
            
            if self.__empty_cuda_cache_after_scoring:
                lamorel_logger.debug(f"Emptying CUDA cache on process {self.accelerator.process_index}")
                torch.cuda.empty_cache()
            
            lamorel_logger.debug(f"Scoring finished on process {self.accelerator.process_index}")

            return _forward_results


    def forward(self, module_function_keys, contexts, candidates=None, require_grad=False, minibatch_size=None,
                **kwargs):
        """
        This function is implemted here only because it was present in baseLLM.
        vLLM's API doesn't provide any pass forward method.
        """
        raise NotImplementedError("VLLM dosen't provide any pass forward method")