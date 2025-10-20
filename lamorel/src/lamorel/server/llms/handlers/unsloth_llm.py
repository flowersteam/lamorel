from .hf_llm import HF_LLM
import torch
from torch.nn.functional import log_softmax
from peft import PeftModel
from unsloth_zoo.utils import _get_dtype

import time
from unsloth import FastLanguageModel, is_bfloat16_supported

import logging
lamorel_logger = logging.getLogger('lamorel_logger')

class UnslothLLM(HF_LLM):
    """
        This class is a wrapper around the language model, and handles distributing
        and/or compiling the model.
        For each task the default is text in, text out.
        """

    def __init__(self, args, devices, process_index, use_cpu):
        assert not use_cpu and devices == [0] and args.model_type == "causal"
        super().__init__(args, devices, process_index, use_cpu)

    def _instantiate_model(self, model_args):
        print("Launching Unsloth LLM on {} device".format(self.main_device))
        # Load model and tokenizer
        constructor_additional_args = {}
        if model_args.constructor_kwargs is not None:
            constructor_additional_args.update(model_args.constructor_kwargs)

        self._LLM_model, self._LLM_tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_args.model_path,
            dtype=None,
            load_in_4bit=model_args.load_in_4bit,
            **constructor_additional_args
        )

    def generate(self, contexts, return_logprobs=False, peft_adapter=None, **kwargs):
        if contexts is None or len(contexts) == 0:
            return [{}]

        if isinstance(self._LLM_model, PeftModel) and peft_adapter is not None:
            lamorel_logger.debug(f"Activating {peft_adapter} adapters.")
            self._LLM_model.set_adapter(peft_adapter)

        lamorel_logger.info(
            "Unsloth fast generation is currently disabled as it breaks gradient checkpointing for training with DDP.")
        generations = []
        encoded_inputs = self._LLM_tokenizer(contexts, return_tensors='pt', padding=True, truncation=False,
                                             add_special_tokens=False).to(self.main_device)

        # Don't use fast generate for now as it breaks gradient checkpointing for training with DDP
        # Hence took code from here: https://github.com/unslothai/unsloth/blob/8a055402a27c3d9643cc16947ce40311a280e69c/unsloth/models/llama.py#L1537
        kwargs["cache_implementation"] = "dynamic"
        # For num_logits_to_keep
        # num_logits_to_keep = kwargs.get("num_logits_to_keep", None)
        # logits_to_keep = kwargs.get("logits_to_keep", None)
        # if num_logits_to_keep is None and logits_to_keep is None:
        #     kwargs["num_logits_to_keep"] = 1
        # Remove token_type_ids
        kwargs.pop("token_type_ids", None)
        kwargs["pad_token_id"] = self._LLM_tokenizer.pad_token_id
        dtype = _get_dtype(self._LLM_model.config.torch_dtype)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=dtype):
            results = self._LLM_model._old_generate(
                input_ids=encoded_input["input_ids"],
                attention_mask=encoded_input["attention_mask"],
                return_dict_in_generate=True,
                output_scores=True,
                tokenizer=self._LLM_tokenizer,
                **kwargs
            )

        num_return_sequences = kwargs.get('num_return_sequences', 1)
        batch_size = encoded_inputs["input_ids"].shape[0]
        if self.model_type == "causal":
            generated_sequences = results.sequences[:, encoded_inputs["input_ids"].shape[-1]:]
        else:
            generated_sequences = results.sequences[:, 1:]

        if return_logprobs:
            logp = log_softmax(torch.stack(results.scores, dim=1), dim=-1)
            scores = torch.gather(logp, 2, generated_sequences[:, :, None]).squeeze(-1)
            aggregated_scores = scores.sum(-1)
        else:
            probabilities = torch.stack(results.scores, dim=1).softmax(-1)
            scores = torch.gather(probabilities, 2, generated_sequences[:, :, None]).squeeze(-1)
            aggregated_scores = scores.prod(-1)

        for i in range(batch_size):
            prompt_generations = []
            for j in range(num_return_sequences):
                idx = i * num_return_sequences + j
                _text = self._LLM_tokenizer.decode(generated_sequences[idx], skip_special_tokens=True)
                _tokens = generated_sequences[idx]
                _scores = scores[idx]
                _agg_score = aggregated_scores[idx]
                result = {
                    "text": _text,
                    "tokens": _tokens,
                    "text_probability" if not return_logprobs else "text_logprob": _agg_score.detach().cpu().numpy(),
                    "tokens_probability" if not return_logprobs else "tokens_logprob": _scores.detach().cpu().numpy()
                }
                if "sequences_scores" in results:
                    result["beam_score"] = results.sequences_scores[idx].detach().cpu().item()
                prompt_generations.append(result)

            generations.append(prompt_generations)

        return generations