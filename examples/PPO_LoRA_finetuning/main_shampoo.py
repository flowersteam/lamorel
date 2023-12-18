'''
PPO implementation taken from https://github.com/openai/spinningup
'''

from collections import OrderedDict
from typing import List
from itertools import chain

import hydra
from utils.ppo_buffer import PPOBuffer
from utils.generate_prompt import generate_prompt
from utils.scoring_utils import scores_stacking
import torch
from torch.nn.functional import log_softmax
import numpy as np
import logging

from tqdm import tqdm
import time
import pickle
import math
import os
import functools as f
from operator import add

from transformers import set_seed, top_k_top_p_filtering
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from babyai_text_env import BabyAITextEnv
from shampoo_text_env import ShampooTextEnv

from lamorel import Caller, lamorel_init
from lamorel import BaseUpdater, BaseModuleFunction, BaseModelInitializer

lamorel_init()

from accelerate import Accelerator
accelerator = Accelerator()

class LogScoringModuleFn(BaseModuleFunction):
    def __init__(self, model_type, pre_encoded_input):
        super().__init__()
        self._model_type = model_type
        self._pad_token = 0
        self._pre_encoded_input = pre_encoded_input

    def initialize(self):
        pass

    def forward(self, forward_outputs, minibatch, tokenized_contexts, **kwargs):
        if self._model_type == "causal":
            if self._pre_encoded_input:
                end_of_context_position = 0
            else:  # hence input should be removed from result
                end_of_context_position = len(
                    tokenized_contexts[0]["input_ids"]) # inputs are padded so all of same size

            logits = forward_outputs["logits"][:, end_of_context_position:-1, :]
            output_tokens = minibatch["input_ids"][:, end_of_context_position+1:]
        else:
            logits = forward_outputs["logits"][:, :-1, :]  # skip </s> token appended by tokenizer
            output_tokens = minibatch["decoder_input_ids"][:, 1:]  # skip pad token

        tokens_logprobs = \
            torch.gather(logits, 2, output_tokens[:, :, None]).squeeze(-1).to(torch.float32)  # filter with sequence tokens

        # Compute mask to assign probability 1 to padding tokens
        mask = torch.ones(tokens_logprobs.shape, dtype=torch.bool, device=self.device)
        for i, _output in enumerate(output_tokens):
            for j, _token in enumerate(_output):
                if _token != self._pad_token:
                    mask[i, j] = False
        masked_token_probs = tokens_logprobs.masked_fill(mask, 0.0)  # apply mask
        minibatch_probs = masked_token_probs.sum(-1)  # compute final sequences' probability

        return minibatch_probs.cpu()
    
class LogProbModuleFn(BaseModuleFunction):
    def __init__(self, model_type, pre_encoded_input):
        super().__init__()
        self._model_type = model_type
        self._pad_token = 0
        self._pre_encoded_input = pre_encoded_input

    def initialize(self):
        pass

    def forward(self, forward_outputs, minibatch, tokenized_contexts, **kwargs):
        if self._model_type == "causal":
            if self._pre_encoded_input:
                end_of_context_position = 0
            else:  # hence input should be removed from result
                end_of_context_position = len(
                    tokenized_contexts[0]["input_ids"]) # inputs are padded so all of same size

            logits = forward_outputs["logits"][:, end_of_context_position:-1, :]
            output_tokens = minibatch["input_ids"][:, end_of_context_position+1:]
        else:
            logits = forward_outputs["logits"][:, :-1, :]  # skip </s> token appended by tokenizer
            output_tokens = minibatch["decoder_input_ids"][:, 1:]  # skip pad token

        tokens_logprobs = \
            torch.gather(logits, 2, output_tokens[:, :, None]).squeeze(-1).to(torch.float32)  # filter with sequence tokens

        # Compute mask to assign probability 1 to padding tokens
        mask = torch.ones(tokens_logprobs.shape, dtype=torch.bool, device=self.device)
        for i, _output in enumerate(output_tokens):
            for j, _token in enumerate(_output):
                if _token != self._pad_token:
                    mask[i, j] = False
        masked_token_probs = tokens_logprobs.masked_fill(mask, -np.inf)  # apply mask
        masked_token_probs = masked_token_probs.log_softmax(-1)
        masked_token_probs = masked_token_probs.masked_fill(mask, 0.0)  # apply mask again
        minibatch_probs = masked_token_probs.sum(-1)  # compute final sequences' log probability

        return minibatch_probs.cpu()

class ValueHeadModuleFn(BaseModuleFunction):
    def __init__(self, model_type, pre_encoded_input):
        super().__init__()
        self._model_type = model_type
        self._pre_encoded_input = pre_encoded_input

    def initialize(self):
        if 'hidden_size' in self.llm_config.attribute_map:
            _hidden_size_key = self.llm_config.attribute_map['hidden_size']
        else:
            if "word_embed_proj_dim" in self.llm_config.to_dict():
                _hidden_size_key = "word_embed_proj_dim"
            elif "hidden_size" in self.llm_config.to_dict():
                _hidden_size_key = "hidden_size"
            else:
                print(self.llm_config.to_dict())
                raise NotImplementedError("Unknown hidden size key")

        self._llm_hidden_size = self.llm_config.to_dict()[_hidden_size_key]
        self.value_head_op = torch.nn.Sequential(
            torch.nn.Linear(self._llm_hidden_size, 1024),
            torch.nn.Sigmoid(),
            torch.nn.Linear(1024, 1024),
            torch.nn.Sigmoid(),
            torch.nn.Linear(1024, 1),
        ).to(self.device)

    def forward(self, forward_outputs, minibatch, tokenized_contexts, **kwargs):
        # Get last layer's hidden from last token in context
        if self._model_type == "causal":
            if self._pre_encoded_input:
                end_of_context_position = 0
            else:  # hence input should be removed from result
                end_of_context_position = len(
                    tokenized_contexts[0]["input_ids"])  # inputs are padded so all of same size

            model_head = forward_outputs['hidden_states'][-1][:, end_of_context_position, :]
        else:
            model_head = forward_outputs["decoder_hidden_states"][-1][:, 0, :]

        value = self.value_head_op(model_head.to(torch.float32).to(self.device))
        return value.cpu()

class SequentialInitializer(BaseModelInitializer):
    def __init__(self, initializers:List[BaseModelInitializer]):
        super().__init__()
        self._initializers = initializers

    def initialize_model(self, model):
        for _initializer in self._initializers:
            model = _initializer.initialize_model(model)

        return model

class WeightsLoaderInitializer(BaseModelInitializer):
    def __init__(self, weights_path):
        super().__init__()
        self._weights_path = weights_path

    def initialize_model(self, model):
        if self._weights_path is not None:
            loaded_ddp_dict = torch.load(self._weights_path + "/model.checkpoint")
            hf_llm_module_dict = {_k.replace('module.', ''): _v for _k, _v in loaded_ddp_dict.items()}
            model.load_state_dict(state_dict=hf_llm_module_dict, strict=True)

        return model

class PeftInitializer(BaseModelInitializer):
    def __init__(self, model_type, model_name, use_lora, use_4bit, r, alpha, use_cache=True):
        super().__init__()
        self._model_type = model_type
        self._model_name = model_name
        self._use_lora = use_lora
        self._use_4bit = use_4bit
        self._r = r
        self._alpha = alpha
        self._use_cache = use_cache

    def _print_trainable_parameters(self, model):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )

    def _get_model_config(self):
        if "t5" in self._model_name:
            return LoraConfig(
                r=self._r,
                lora_alpha=self._alpha,
                target_modules=["q", "v"],
                lora_dropout=0.0,
                bias="none",
                task_type="SEQ_2_SEQ_LM"
            )
        elif "opt" in self._model_name or "Llama" in self._model_name or "Mistral" in self._model_name:
            return LoraConfig(
                r=self._r,
                lora_alpha=self._alpha,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.0,
                bias="none",
                task_type="CAUSAL_LM"
            )
        else:
            raise NotImplementedError()

    def initialize_model(self, model):
        if self._use_lora:
            llm_module = model._modules['_LLM_model']
            if self._model_type == "seq2seq" or not self._use_cache:
                llm_module.gradient_checkpointing_enable()  # reduce number of stored activations

            if self._use_4bit:
                llm_module = prepare_model_for_kbit_training(llm_module)

            # Init adapters #
            config = self._get_model_config()
            peft_model = get_peft_model(llm_module, config)
            parent_module_device = None
            for name, param in peft_model.named_modules():
                if name.split(".")[-1].startswith("lora_"):
                    if hasattr(param, "weight"):
                        param.to(parent_module_device)
                else:
                    if hasattr(param, "weight"):
                        parent_module_device = param.weight.device
                    else:
                        parent_module_device = None

            model._modules['_LLM_model'] = peft_model

        model.eval()  # Important to ensure ratios are 1 in first minibatch of PPO (i.e. no dropout)
        model._modules['_LLM_model'].config.use_cache = self._use_cache
        self._print_trainable_parameters(model)
        return model

class PPOUpdater(BaseUpdater):
    def __init__(self, model_type, minibatch_size, gradient_batch_size, gradient_minibatch_size=None):
        super(PPOUpdater, self).__init__()
        self._model_type = model_type
        self._minibatch_size = minibatch_size
        self._gradient_batch_size = gradient_batch_size
        self._gradient_minibatch_size = gradient_minibatch_size

    def _get_trainable_params(self, model, return_with_names=False):
        if return_with_names:
            return filter(lambda p: p[1].requires_grad, model.named_parameters())
        else:
            return filter(lambda p: p.requires_grad, model.parameters())

    def perform_update(self, contexts, candidates, _current_batch_ids, **kwargs):
        # import pdb; pdb.set_trace()
        if not hasattr(self, 'optimizer'):
            self._iterator_named_trainable_params = lambda: self._get_trainable_params(self._llm_module, True)
            self._iterator_trainable_params = (p for n, p in self._iterator_named_trainable_params())
            self.optimizer = torch.optim.Adam(self._iterator_trainable_params, lr=kwargs["lr"])

            if os.path.exists(kwargs["loading_path"] + "/optimizer.checkpoint"):
                self.optimizer.load_state_dict(torch.load(kwargs["loading_path"] + "/optimizer.checkpoint"))

        if kwargs['agent'] == 'seller':
            current_process_buffer = {}
            for k in ['actions', 'advantages', 'returns', 'logprobs', 'values']:
                current_process_buffer[k] = kwargs[k][_current_batch_ids]

            epochs_losses = {
                "value_seller": [],
                "policy_seller": [],
                "loss_seller": []
            }

            n_minibatches = math.ceil(len(contexts) / self._minibatch_size)
            for i in tqdm(range(kwargs["ppo_epochs"]), ascii=" " * 9 + ">", ncols=100):
                for step in range(n_minibatches):
                    _minibatch_start_idx = step * self._minibatch_size
                    _minibatch_end_idx = min(
                        (step + 1) * self._minibatch_size,
                        len(contexts))
                    
                    self.optimizer.zero_grad()
                    gradient_accumulation_steps = math.ceil(
                        _minibatch_end_idx - _minibatch_start_idx / self._gradient_batch_size)
                    for accumulated_batch in range(gradient_accumulation_steps):
                        _start_idx = _minibatch_start_idx + accumulated_batch * self._gradient_batch_size
                        _stop_idx = _minibatch_start_idx + min(
                            (accumulated_batch + 1) * self._gradient_batch_size, _minibatch_end_idx)

                        _contexts = contexts[_start_idx:_stop_idx]
                        _candidates = candidates[_start_idx:_stop_idx]
                        if len(_contexts) == 0: break
                        if self._gradient_minibatch_size is None:
                            _batch_size = sum(len(_c) for _c in _candidates)
                        else:
                            _batch_size = self._gradient_minibatch_size
                        # Use LLM to compute again action probabilities and value
                        output = self._llm_module(
                            ['score_seller', 'value_seller'], 
                            contexts=_contexts, 
                            candidates=_candidates,
                            require_grad=True, 
                            minibatch_size=_batch_size)
                        log_probs = torch.stack([_o['score_seller'] for _o in output]).squeeze()
                        values = torch.stack([_o["value_seller"][0] for _o in output]).squeeze()

                        # Compute policy loss
                        # TODO: how to calculate the entropy for a large language model?
                        # does it make sense to calculate the entropy for a large language model?
                        # does it need to calculate the entropy for a large language model?
                        # entropy = probas.entropy().mean()
                        if len(log_probs.shape) == 0:
                            log_probs = log_probs.unsqueeze(0)
                        # log_prob = log_probs[_start_idx:_stop_idx] # Use logprobs from dist as they were normalized
                        log_prob = log_probs
                        ratio = torch.exp(log_prob - current_process_buffer['logprobs'][_start_idx:_stop_idx])
                        # assert not (i == 0 and step == 0 and (torch.any(ratio < 0.99) or torch.any(ratio > 1.1)))
                        if i == 0 and step == 0 and (torch.any(ratio < 0.99) or torch.any(ratio > 1.1)):
                            logging.warning("PPO ratio != 1 !!")

                        clip_adv = torch.clamp(ratio, 1 - kwargs["clip_eps"], 1 + kwargs["clip_eps"]) * current_process_buffer['advantages'][_start_idx:_stop_idx]
                        policy_loss = -(torch.min(ratio * current_process_buffer['advantages'][_start_idx:_stop_idx], clip_adv)).mean()
                        epochs_losses["policy_seller"].append(policy_loss.detach().cpu().item())

                        # Compute value loss
                        unclipped_value_error = ((values - current_process_buffer['returns'][_start_idx:_stop_idx]) ** 2)
                        clipped_values = current_process_buffer['values'][_start_idx:_stop_idx] + \
                                        torch.clamp(values - current_process_buffer['values'][_start_idx:_stop_idx],
                                                    -kwargs["clip_eps"], kwargs["clip_eps"])
                        clipped_value_error = ((clipped_values - current_process_buffer['returns'][_start_idx:_stop_idx]) ** 2)
                        value_loss = torch.max(unclipped_value_error, clipped_value_error).mean()
                        epochs_losses["value_seller"].append(value_loss.detach().cpu().item())

                        # Compute final loss
                        # loss = policy_loss - kwargs["entropy_coef"] * entropy + kwargs["value_loss_coef"] * value_loss
                        loss = policy_loss + kwargs["value_loss_coef"] * value_loss
                        loss = loss / gradient_accumulation_steps
                        epochs_losses["loss_seller"].append(loss.detach().cpu().item())

                        # Backward
                        loss.backward()
                    torch.nn.utils.clip_grad_norm_(self._iterator_trainable_params, kwargs["max_grad_norm"])
                    self.optimizer.step()

            if kwargs["save_after_update"] and accelerator.process_index == 1:
                print("Saving seller's model...")
                model_state_dict = OrderedDict({
                        k: v for k, v in self._iterator_named_trainable_params()
                    })
                torch.save(model_state_dict, kwargs["output_dir"] + "/model.checkpoint")
                torch.save(self.optimizer.state_dict(), kwargs["output_dir"] + "/optimizer.checkpoint")
                print("Seller's model saved")

            return {'loss_seller': np.mean(epochs_losses["loss_seller"]), 'value_loss_seller': np.mean(epochs_losses["value_seller"]),
                    'policy_loss_seller': np.mean(epochs_losses["policy_seller"])}

        elif kwargs['agent'] == 'buyer':
            current_process_buffer = {}
            for k in ['actions', 'advantages', 'returns', 'logprobs', 'values']:
                current_process_buffer[k] = kwargs[k][_current_batch_ids]

            epochs_losses = {
                "value_buyer": [],
                "policy_buyer": [],
                "loss_buyer": []
            }

            n_minibatches = math.ceil(len(contexts) / self._minibatch_size)
            for i in tqdm(range(kwargs["ppo_epochs"]), ascii=" " * 9 + ">", ncols=100):
                for step in range(n_minibatches):
                    _minibatch_start_idx = step * self._minibatch_size
                    _minibatch_end_idx = min(
                        (step + 1) * self._minibatch_size,
                        len(contexts))

                    self.optimizer.zero_grad()
                    gradient_accumulation_steps = math.ceil(
                        _minibatch_end_idx - _minibatch_start_idx / self._gradient_batch_size)
                    for accumulated_batch in range(gradient_accumulation_steps):
                        _start_idx = _minibatch_start_idx + accumulated_batch * self._gradient_batch_size
                        _stop_idx = _minibatch_start_idx + min(
                            (accumulated_batch + 1) * self._gradient_batch_size, _minibatch_end_idx)

                        _contexts = contexts[_start_idx:_stop_idx]
                        _candidates = candidates[_start_idx:_stop_idx]
                        if len(_contexts) == 0: break
                        if self._gradient_minibatch_size is None:
                            _batch_size = sum(len(_c) for _c in _candidates)
                        else:
                            _batch_size = self._gradient_minibatch_size
                        # Use LLM to compute again action probabilities and value
                        output = self._llm_module(['score_buyer', 'value_buyer'], contexts=_contexts, candidates=_candidates,
                                                require_grad=True, minibatch_size=_batch_size)
                        scores = torch.stack([_o['score_buyer'] for _o in output]).squeeze()
                        probas = torch.distributions.Categorical(logits=scores)
                        values = torch.stack([_o["value_buyer"][0] for _o in output]).squeeze()

                        # Compute policy loss
                        entropy = probas.entropy().mean()
                        log_prob = probas.log_prob(current_process_buffer['actions'][_start_idx:_stop_idx]) # Use logprobs from dist as they were normalized
                        ratio = torch.exp(log_prob - current_process_buffer['logprobs'][_start_idx:_stop_idx])
                        # assert not (i == 0 and step == 0 and (torch.any(ratio < 0.99) or torch.any(ratio > 1.1)))
                        if i == 0 and step == 0 and (torch.any(ratio < 0.99) or torch.any(ratio > 1.1)):
                            logging.warning("PPO ratio != 1 !!")

                        clip_adv = torch.clamp(ratio, 1 - kwargs["clip_eps"], 1 + kwargs["clip_eps"]) * current_process_buffer['advantages'][_start_idx:_stop_idx]
                        policy_loss = -(torch.min(ratio * current_process_buffer['advantages'][_start_idx:_stop_idx], clip_adv)).mean()
                        epochs_losses["policy_buyer"].append(policy_loss.detach().cpu().item())

                        # Compute value loss
                        unclipped_value_error = ((values - current_process_buffer['returns'][_start_idx:_stop_idx]) ** 2)
                        clipped_values = current_process_buffer['values'][_start_idx:_stop_idx] + \
                                        torch.clamp(values - current_process_buffer['values'][_start_idx:_stop_idx],
                                                    -kwargs["clip_eps"], kwargs["clip_eps"])
                        clipped_value_error = ((clipped_values - current_process_buffer['returns'][_start_idx:_stop_idx]) ** 2)
                        value_loss = torch.max(unclipped_value_error, clipped_value_error).mean()
                        epochs_losses["value_buyer"].append(value_loss.detach().cpu().item())

                        # Compute final loss
                        loss = policy_loss - kwargs["entropy_coef"] * entropy + kwargs["value_loss_coef"] * value_loss
                        loss = loss / gradient_accumulation_steps
                        epochs_losses["loss_buyer"].append(loss.detach().cpu().item())

                        # Backward
                        loss.backward()
                    torch.nn.utils.clip_grad_norm_(self._iterator_trainable_params, kwargs["max_grad_norm"])
                    self.optimizer.step()

            if kwargs["save_after_update"] and accelerator.process_index == 1:
                print("Saving buyer's model...")
                model_state_dict = OrderedDict({
                        k: v for k, v in self._iterator_named_trainable_params()
                    })
                torch.save(model_state_dict, kwargs["output_dir"] + "/model.checkpoint")
                torch.save(self.optimizer.state_dict(), kwargs["output_dir"] + "/optimizer.checkpoint")
                print("Buyer's model saved")

            return {'loss_buyer': np.mean(epochs_losses["loss_buyer"]), 'value_loss_buyer': np.mean(epochs_losses["value_buyer"]),
                    'policy_loss_buyer': np.mean(epochs_losses["policy_buyer"])}

def reset_history():
    return {
        "ep_len": [],
        "ep_ret": [],
        "loss": [],
        "policy_loss": [],
        "value_loss": [],
        "possible_actions": [],
        "actions": [],
        "prompts": [],
    }

@hydra.main(config_path='config', config_name='config')
def main(config_args):
    # Random seed
    seed = config_args.rl_script_args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    set_seed(seed)

    # Instantiate environment
    # envs = BabyAITextEnv(config_args.rl_script_args)
    envs = ShampooTextEnv(config_args.rl_script_args)

    # Create LLM agent
    lm_server = Caller(config_args.lamorel_args,
                       custom_updater=PPOUpdater(config_args.lamorel_args.llm_args.model_type,
                                                 config_args.rl_script_args.minibatch_size,
                                                 config_args.rl_script_args.gradient_batch_size),
                       custom_model_initializer=SequentialInitializer([
                           PeftInitializer(config_args.lamorel_args.llm_args.model_type,
                                           config_args.lamorel_args.llm_args.model_path,
                                           config_args.rl_script_args.use_lora,
                                           config_args.lamorel_args.llm_args.load_in_4bit,
                                           config_args.rl_script_args.lora_r,
                                           config_args.rl_script_args.lora_alpha,
                                           config_args.lamorel_args.llm_args.pre_encode_inputs),
                           WeightsLoaderInitializer(config_args.rl_script_args.loading_path)
                       ]),
                       custom_module_functions={
                           'score_seller': LogProbModuleFn(
                               config_args.lamorel_args.llm_args.model_type,
                               config_args.lamorel_args.llm_args.pre_encode_inputs),
                           'value_seller': ValueHeadModuleFn(
                               config_args.lamorel_args.llm_args.model_type,
                               config_args.lamorel_args.llm_args.pre_encode_inputs),
                           'score_buyer': LogScoringModuleFn(
                               config_args.lamorel_args.llm_args.model_type,
                               config_args.lamorel_args.llm_args.pre_encode_inputs),
                           'value_buyer': ValueHeadModuleFn(
                               config_args.lamorel_args.llm_args.model_type,
                               config_args.lamorel_args.llm_args.pre_encode_inputs),
                       })

    # Set up experience buffer for the buyer
    buffers_buyer = [
        PPOBuffer(config_args.rl_script_args.steps_per_epoch // config_args.rl_script_args.number_envs,
                  config_args.rl_script_args.gamma, config_args.rl_script_args.lam)
        for _ in range(config_args.rl_script_args.number_envs)
    ]
    # Set up experience buffer for the seller
    buffers_seller = [
        PPOBuffer(config_args.rl_script_args.steps_per_epoch // config_args.rl_script_args.number_envs,
                  config_args.rl_script_args.gamma, config_args.rl_script_args.lam)
        for _ in range(config_args.rl_script_args.number_envs)
    ]

    history_buyer = reset_history()
    history_seller = reset_history()
    # TODO: check whether the goal is needed
    # history["goal"].extend([_i["goal"] for _i in infos])

    for epoch in range(config_args.rl_script_args.epochs):

        __time = time.time()
        
        for t in tqdm(range(config_args.rl_script_args.steps_per_epoch // config_args.rl_script_args.number_envs),
                      ascii=" " * 9 + ">", ncols=100):
            # Prepare for interaction with environment
            (o, infos), _ret, _len = envs.reset(
                seeds=[config_args.rl_script_args.seed+config_args.rl_script_args.number_envs*(epoch+1)*t+i for i in range(config_args.rl_script_args.number_envs)]), \
                [0 for _ in range(config_args.rl_script_args.number_envs)], \
                [0 for _ in range(config_args.rl_script_args.number_envs)]
            ep_ret_seller, ep_ret_buyer = _ret.copy(), _ret.copy()
            ep_len_seller, ep_len_buyer = _len.copy(), _len.copy()

            # seller step
            prompts_seller = [_i['seller']['prompt'] for _i in infos]
            generations = lm_server.generate(
                contexts=prompts_seller,
                temperature=config_args.lamorel_args.llm_args.temperature,
                max_length=config_args.lamorel_args.llm_args.max_length,
                do_sample=True,
                return_logprobs=True,)
            actions_seller = [[_o[0]['text'][:config_args.lamorel_args.llm_args.max_length]] for _o in generations]
            output_seller = lm_server.custom_module_fns(['score_seller', 'value_seller'],
                                                 contexts=prompts_seller,
                                                 candidates=actions_seller)
            log_probs_seller = torch.stack([_o['score_seller'] for _o in output_seller])
            values_seller = torch.stack([_o["value_seller"][0] for _o in output_seller])
            o, r, d, infos = envs.step(list(chain.from_iterable(actions_seller)))

            # buyer step
            possible_actions_buyer = [config_args.rl_script_args.buyer_action_space for _ in range(config_args.rl_script_args.number_envs)]
            prompts_buyer = [_i['buyer']['prompt'] for _i in infos]
            output_buyer = lm_server.custom_module_fns(
                ['score_buyer', 'value_buyer'], contexts=prompts_buyer,
                candidates=possible_actions_buyer)
            scores_buyer = scores_stacking([_o['score_buyer'] for _o in output_buyer])
            proba_dist_buyer = torch.distributions.Categorical(logits=scores_buyer)
            values_buyer = torch.stack([_o["value_buyer"][0] for _o in output_buyer])
            sampled_actions = proba_dist_buyer.sample()
            log_probs_buyer = proba_dist_buyer.log_prob(sampled_actions)
            actions_id = sampled_actions.cpu().numpy()
            actions_buyer = []
            for j in range(len(actions_id)):
                command = possible_actions_buyer[j][int(actions_id[j])]
                actions_buyer.append(int(command))

            o, r, d, infos = envs.step(actions_buyer)

            epoch_ended = (t+1)*config_args.rl_script_args.number_envs == config_args.rl_script_args.steps_per_epoch
            bootstrap_dict = {
                "ids": [],
                "contexts": []
            }
            for i in range(config_args.rl_script_args.number_envs):
                buffers_seller[i].store(prompts_seller[i], actions_seller[i], None, r[i]['seller'], values_seller[i], log_probs_seller[i])
                buffers_buyer[i].store(prompts_buyer[i], possible_actions_buyer[i], actions_id[i], r[i]['buyer'], values_buyer[i], log_probs_buyer[i])
                ep_ret_seller[i] += r[i]['seller']
                ep_ret_buyer[i] += r[i]['buyer']
                ep_len_seller[i] += 1
                ep_len_buyer[i] += 1
                timeout = (ep_len_seller[i] == config_args.rl_script_args.max_ep_len) and (ep_len_buyer[i] == config_args.rl_script_args.max_ep_len)
                terminal = d[i] or timeout
                if terminal or epoch_ended:
                    if not terminal:
                        # bootstrap_dict["ids"].append(i)
                        # bootstrap_dict["contexts"].append(generate_prompt(o[i], infos[i]))
                        raise NotImplementedError("Bootstrap not implemented")
                    else:
                        buffers_seller[i].finish_path(0)
                        buffers_buyer[i].finish_path(0)
                        history_seller["ep_len"].append(ep_len_seller[i])
                        history_buyer["ep_len"].append(ep_len_buyer[i])
                        history_seller["ep_ret"].append(ep_ret_seller[i])
                        history_buyer["ep_ret"].append(ep_len_buyer[i])
                        # ep_len_seller[i], ep_ret_seller[i] = 0, 0
                        # ep_len_buyer[i], ep_ret_buyer[i] = 0, 0

            # the following code will not be executed in the current version
            if len(bootstrap_dict["ids"]) > 0:
                raise NotImplementedError

        # Perform PPO update!
        print(f"PPO update number {epoch + 1}")
        save_model_and_history = (epoch % config_args.rl_script_args.save_freq == 0 or
                                  epoch == config_args.rl_script_args.epochs - 1) and epoch != 0
        start_epoch = epoch - config_args.rl_script_args.save_freq
        saving_path_seller = f"{config_args.rl_script_args.output_dir}/seller_epochs_{start_epoch}-{epoch}"
        saving_path_buyer = f"{config_args.rl_script_args.output_dir}/buyer_epochs_{start_epoch}-{epoch}"
        if save_model_and_history:
            os.makedirs(saving_path_seller, exist_ok=True)
            os.makedirs(saving_path_buyer, exist_ok=True)
        loading_path = config_args.rl_script_args.loading_path \
            if config_args.rl_script_args.loading_path is not None else ""
        
        # import pdb; pdb.set_trace()

        # Stack trajectories for all envs
        # TODO: Randomize and mix up environments' trajectories
        trajectories_seller = [buf.get() for buf in buffers_seller]
        trajectories_buyer = [buf.get() for buf in buffers_buyer]
        collected_trajectories_seller = {
            k: torch.cat([traj[k] for traj in trajectories_seller]) if isinstance(trajectories_seller[0][k], torch.Tensor)
            else list(f.reduce(add, [traj[k] for traj in trajectories_seller]))
            for k, _ in trajectories_seller[0].items()
        }
        collected_trajectories_buyer = {
            k: torch.cat([traj[k] for traj in trajectories_buyer]) if isinstance(trajectories_buyer[0][k], torch.Tensor)
            else list(f.reduce(add, [traj[k] for traj in trajectories_buyer]))
            for k, _ in trajectories_buyer[0].items()
        }

        # seller_update
        update_results_seller = lm_server.update(
            collected_trajectories_seller['obs'],
            collected_trajectories_seller['possible_act'],
            agent='seller',
            actions=collected_trajectories_seller['act'],
            returns=collected_trajectories_seller['ret'],
            advantages=collected_trajectories_seller['adv'],
            logprobs=collected_trajectories_seller['logp'],
            values=collected_trajectories_seller['val'],
            lr=config_args.rl_script_args.lr,
            clip_eps=config_args.rl_script_args.clip_eps,
            entropy_coef=config_args.rl_script_args.entropy_coef,
            value_loss_coef=config_args.rl_script_args.value_loss_coef,
            max_grad_norm=config_args.rl_script_args.max_grad_norm,
            ppo_epochs=config_args.rl_script_args.ppo_epochs,
            save_after_update=save_model_and_history,
            output_dir=saving_path_seller,
            loading_path=loading_path
        )

        avg_loss = np.mean([_r['loss_seller'] for _r in update_results_seller])
        avg_policy_loss = np.mean([_r['policy_loss_seller'] for _r in update_results_seller])
        avg_value_loss = np.mean([_r['value_loss_seller'] for _r in update_results_seller])
        history_seller["loss"].append(avg_loss)
        history_seller["policy_loss"].append(avg_policy_loss)
        history_seller["value_loss"].append(avg_value_loss)
        history_seller["possible_actions"].extend(collected_trajectories_seller['possible_act'])
        # history_seller["actions"].extend([
        #     _poss_act[int(_a.item())] for _poss_act, _a in
        #     zip(collected_trajectories_seller['possible_act'], collected_trajectories_seller['act'])])
        history_seller["prompts"].extend(collected_trajectories_seller['obs'])
        print(f"Update seller's loss: {avg_loss}")

        if save_model_and_history:
            # Save history
            with open(f"{saving_path_seller}/history.pkl", "wb") as file:
                pickle.dump(history_seller, file)
            history_seller = reset_history()

        # buyer_update
        update_results_buyer = lm_server.update(
            collected_trajectories_buyer['obs'],
            collected_trajectories_buyer['possible_act'],
            agent='buyer',
            actions=collected_trajectories_buyer['act'],
            returns=collected_trajectories_buyer['ret'],
            advantages=collected_trajectories_buyer['adv'],
            logprobs=collected_trajectories_buyer['logp'],
            values=collected_trajectories_buyer['val'],
            lr=config_args.rl_script_args.lr,
            clip_eps=config_args.rl_script_args.clip_eps,
            entropy_coef=config_args.rl_script_args.entropy_coef,
            value_loss_coef=config_args.rl_script_args.value_loss_coef,
            max_grad_norm=config_args.rl_script_args.max_grad_norm,
            ppo_epochs=config_args.rl_script_args.ppo_epochs,
            save_after_update=save_model_and_history,
            output_dir=saving_path_buyer,
            loading_path=loading_path
        )

        avg_loss = np.mean([_r['loss_buyer'] for _r in update_results_buyer])
        avg_policy_loss = np.mean([_r['policy_loss_buyer'] for _r in update_results_buyer])
        avg_value_loss = np.mean([_r['value_loss_buyer'] for _r in update_results_buyer])
        history_buyer["loss"].append(avg_loss)
        history_buyer["policy_loss"].append(avg_policy_loss)
        history_buyer["value_loss"].append(avg_value_loss)
        history_buyer["possible_actions"].extend(collected_trajectories_buyer['possible_act'])
        history_buyer["actions"].extend([
            _poss_act[int(_a.item())] for _poss_act, _a in
            zip(collected_trajectories_buyer['possible_act'], collected_trajectories_buyer['act'])])
        history_buyer["prompts"].extend(collected_trajectories_buyer['obs'])
        print(f"Update buyer's loss: {avg_loss}")

        if save_model_and_history:
            # Save history
            with open(f"{saving_path_buyer}/history.pkl", "wb") as file:
                pickle.dump(history_buyer, file)
            history_buyer = reset_history()

    start_epoch = epoch - config_args.rl_script_args.save_freq
    # save seller's history
    saving_path_seller = f"{config_args.rl_script_args.output_dir}/seller_epochs_{start_epoch}-{epoch}"
    os.makedirs(saving_path_seller, exist_ok=True)
    with open(f"{saving_path_seller}/history.pkl", "wb") as file:
        pickle.dump(history_seller, file)
    # save buyer's history
    saving_path_buyer = f"{config_args.rl_script_args.output_dir}/buyer_epochs_{start_epoch}-{epoch}"
    os.makedirs(saving_path_buyer, exist_ok=True)
    with open(f"{saving_path_buyer}/history.pkl", "wb") as file:
        pickle.dump(history_buyer, file)
    lm_server.close()

if __name__ == '__main__':
    main()