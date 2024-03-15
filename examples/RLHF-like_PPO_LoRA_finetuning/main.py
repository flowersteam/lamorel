'''
PPO implementation taken from https://github.com/openai/spinningup
'''

from collections import OrderedDict
from typing import List

import hydra
from utils.ppo_buffer import PPOBuffer
import torch
from torch.nn.functional import log_softmax
import numpy as np
import logging

from tqdm import tqdm
import time
import pickle
import math
import os

from transformers import set_seed, top_k_top_p_filtering, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from repeat_env import RepeatEnv

from lamorel import Caller, lamorel_init
from lamorel import BaseUpdater, BaseModuleFunction, BaseModelInitializer

lamorel_init()

from accelerate import Accelerator

class LMLogitsModuleFn(BaseModuleFunction):
    def __init__(self, model_type):
        super().__init__()
        self._model_type = model_type
        self._pad_token = 0

    def initialize(self):
        pass

    def forward(self, forward_outputs, minibatch, tokenized_contexts, **kwargs):
        # Get last layer's logits from last token in context
        if self._model_type == "causal":
            logits = forward_outputs["logits"][:, -1, :]
        else:
            logits = forward_outputs["logits"][:, 0, :]  # skip </s> token appended by tokenizer

        return log_softmax(logits, dim=1).cpu()  # ensures we do have log probabilities defined over ]-inf;0]



class ValueHeadModuleFn(BaseModuleFunction):
    def __init__(self, model_type):
        super().__init__()
        self._model_type = model_type

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
            model_head = forward_outputs['hidden_states'][-1][:, -1, :]
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
        if not hasattr(self, '_accelerator'):
            self._accelerator = Accelerator()

        iterator_named_trainable_params = lambda: self._get_trainable_params(self._llm_module, True)
        iterator_trainable_params = (p for n, p in iterator_named_trainable_params())
        if not hasattr(self, 'optimizer'):
            self.optimizer = torch.optim.Adam(iterator_trainable_params, lr=kwargs["lr"])

        current_process_buffer = {}
        for k in ['actions', 'advantages', 'returns', 'logprobs', 'values']:
            current_process_buffer[k] = kwargs[k][_current_batch_ids]

        longest_candidate = max([len(_c) for _c in candidates])
        epochs_losses = {
            "value": [],
            "policy": [],
            "loss": []
        }

        base_log_probs = []
        n_minibatches = math.ceil(len(contexts) / self._minibatch_size)
        for i in tqdm(range(kwargs["ppo_epochs"]), ascii=" " * 9 + ">", ncols=100):
            if isinstance(base_log_probs, list) and len(base_log_probs) > 0:
                base_log_probs = torch.cat(base_log_probs)

            for step in range(n_minibatches):
                _minibatch_start_idx = step * self._minibatch_size
                _minibatch_end_idx = min(
                    (step + 1) * self._minibatch_size,
                    len(contexts))

                self.optimizer.zero_grad()
                gradient_accumulation_steps = math.ceil(
                    (_minibatch_end_idx - _minibatch_start_idx) / self._gradient_batch_size)
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
                    output = self._llm_module(['value', 'lm_head'], contexts=_contexts,
                                              require_grad=True, minibatch_size=_batch_size * longest_candidate)

                    scores = torch.stack([_o['lm_head'].squeeze() for _o in output])
                    probas = torch.distributions.Categorical(logits=scores)
                    values = torch.stack([_o["value"][0] for _o in output]).squeeze()

                    # Compute policy loss
                    entropy = probas.entropy().mean()
                    log_prob = torch.gather(
                        scores,
                        1,
                        torch.unsqueeze(current_process_buffer['actions'][_start_idx:_stop_idx], 1).long()
                    ).squeeze()
                    ratio = torch.exp(log_prob - current_process_buffer['logprobs'][_start_idx:_stop_idx])
                    # assert not (i == 0 and step == 0 and (torch.any(ratio < 0.99) or torch.any(ratio > 1.1))), print(f"{_contexts}\n{ratio}")
                    clip_adv = torch.clamp(ratio, 1 - kwargs["clip_eps"], 1 + kwargs["clip_eps"]) * current_process_buffer['advantages'][_start_idx:_stop_idx]
                    policy_loss = -(torch.min(ratio * current_process_buffer['advantages'][_start_idx:_stop_idx], clip_adv)).mean()
                    epochs_losses["policy"].append(policy_loss.detach().cpu().item())

                    # Compute value loss
                    unclipped_value_error = ((values - current_process_buffer['returns'][_start_idx:_stop_idx]) ** 2)
                    clipped_values = current_process_buffer['values'][_start_idx:_stop_idx] + \
                                     torch.clamp(values - current_process_buffer['values'][_start_idx:_stop_idx],
                                                 -kwargs["clip_eps"], kwargs["clip_eps"])
                    clipped_value_error = ((clipped_values - current_process_buffer['returns'][_start_idx:_stop_idx]) ** 2)
                    value_loss = torch.max(unclipped_value_error, clipped_value_error).mean()
                    epochs_losses["value"].append(value_loss.detach().cpu().item())

                    # Compute final loss
                    loss = policy_loss + kwargs["value_loss_coef"] * value_loss
                    loss -= kwargs["entropy_coef"] * entropy  # entropy bonus
                    if kwargs["kl_coef"] > 0:
                        if i == 0:
                            with self._llm_module.module._modules['_LLM_model'].disable_adapter():
                                base_output = self._llm_module(['lm_head'], contexts=_contexts, require_grad=False,
                                                               minibatch_size=_batch_size * longest_candidate)
                                _base_log_probs = torch.stack([_o['lm_head'] for _o in base_output]).squeeze()
                                _base_log_probs = torch.gather(
                                    _base_log_probs,
                                    1,
                                    torch.unsqueeze(current_process_buffer['actions'][_start_idx:_stop_idx], 1).long()
                                ).squeeze()
                                base_log_probs.append(_base_log_probs)
                        else:
                            _base_log_probs = base_log_probs[_start_idx:_stop_idx]

                        kl_penalty = (log_prob - _base_log_probs)
                        loss += kwargs["kl_coef"] * kl_penalty.mean()  # kl penalty

                    loss = loss / gradient_accumulation_steps
                    epochs_losses["loss"].append(loss.detach().cpu().item())

                    if kwargs['skip_ratio_spikes'] and ratio.mean() > 10.0:
                        print(f"Warning: skipping {accumulated_batch}th minibatch from epoch {i}")
                        loss = loss * 0.0

                    # Backward
                    loss.backward()
                torch.nn.utils.clip_grad_norm_(iterator_trainable_params, kwargs["max_grad_norm"])
                self.optimizer.step()

        if kwargs["save_after_update"] and self._accelerator.process_index == 1:
            print("Saving model...")
            model_state_dict = OrderedDict({
                k: v for k, v in iterator_named_trainable_params()
            })
            torch.save(model_state_dict, kwargs["output_dir"] + "/model.checkpoint")
            torch.save(self.optimizer.state_dict(), kwargs["output_dir"] + "/optimizer.checkpoint")
            print("Model saved")

        return {'loss': np.mean(epochs_losses["loss"]), 'value_loss': np.mean(epochs_losses["value"]),
                'policy_loss': np.mean(epochs_losses["policy"])}

def reset_history():
    return {
        "ep_len": [],
        "ep_ret": [],
        "goal": [],
        "loss": [],
        "policy_loss": [],
        "value_loss": [],
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
    tokenizer = AutoTokenizer.from_pretrained(config_args.lamorel_args.llm_args.model_path)
    env = RepeatEnv(config_args.rl_script_args.strings)

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
                           'lm_head': LMLogitsModuleFn(
                                config_args.lamorel_args.llm_args.model_type
                            ),
                            'value': ValueHeadModuleFn(
                                config_args.lamorel_args.llm_args.model_type
                            )
                       })

    # Set up experience buffer
    buffer = PPOBuffer(config_args.rl_script_args.steps_per_epoch, config_args.rl_script_args.gamma,
                       config_args.rl_script_args.lam)

    def generate(logits, use_sample=True):
        if use_sample:
            logits = top_k_top_p_filtering(logits.unsqueeze(dim=0), top_k=0, top_p=1.0)
            next_token = torch.multinomial(torch.exp(logits), num_samples=1)
        else:
            next_token = np.argmax(logits)

        return next_token.item()

    # Prepare for interaction with environment
    (_o, _info), ep_ret, ep_len = env.reset(), 0, 0
    base_prompt = _o
    history = reset_history()
    tokens_buffer = []
    for epoch in range(config_args.rl_script_args.epochs):
        for t in range(config_args.rl_script_args.steps_per_epoch):
            __time = time.time()
            # Value
            prompt = base_prompt + tokenizer.decode(tokens_buffer)
            output = lm_server.custom_module_fns(['value', 'lm_head'],
                                                 contexts=[prompt],
                                                 minibatch_size=1)[0]
            value = output["value"][0].item()
            logits = output["lm_head"][0]

            a = generate(logits)
            log_prob = logits[a]
            tokens_buffer.append(a)
            decoded_string = tokenizer.decode(tokens_buffer)
            _o, r, d, _info = env.step(decoded_string)
            buffer.store(prompt, a, r, value, log_prob)
            ep_ret += r
            ep_len += 1

            timeout = ep_len == config_args.rl_script_args.max_ep_len
            terminal = d or timeout
            epoch_ended = t == config_args.rl_script_args.steps_per_epoch - 1

            if terminal or epoch_ended:
                if not terminal:
                    print('Warning: trajectory cut off by epoch at %d steps.' % ep_len, flush=True)
                    prompt = base_prompt + tokenizer.decode(tokens_buffer)
                    value = lm_server.custom_module_fns(
                        module_function_keys=['value'],
                        contexts=[prompt],
                        candidates=[[""]],
                        minibatch_size=1
                    )[0]["value"][0].item()
                else:
                    value = 0

                buffer.finish_path(value)
                if terminal:
                    print(f"Ret: {ep_ret}")
                    print(f"Len: {ep_len}")
                    history["ep_len"].append(ep_len)
                    history["ep_ret"].append(ep_ret)
                    (_o, _info), ep_ret, ep_len = env.reset(), 0, 0
                    base_prompt = _o
                    tokens_buffer = []

        # Perform PPO update!
        print(f"PPO update number {epoch + 1}")
        save_model_and_history = (epoch % config_args.rl_script_args.save_freq == 0 or
                                  epoch == config_args.rl_script_args.epochs - 1) and epoch != 0
        start_epoch = epoch - config_args.rl_script_args.save_freq
        saving_path = f"{config_args.rl_script_args.output_dir}/epochs_{start_epoch}-{epoch}"
        if save_model_and_history:
            os.makedirs(saving_path, exist_ok=True)
        loading_path = config_args.rl_script_args.loading_path \
            if config_args.rl_script_args.loading_path is not None else ""


        collected_trajectories = buffer.get()
        update_results = lm_server.update(collected_trajectories['obs'],
                                          [[""] for _ in range(config_args.rl_script_args.steps_per_epoch)],
                                          actions=collected_trajectories['act'],
                                          returns=collected_trajectories['ret'],
                                          advantages=collected_trajectories['adv'],
                                          logprobs=collected_trajectories['logp'],
                                          values=collected_trajectories['val'],
                                          lr=config_args.rl_script_args.lr,
                                          clip_eps=config_args.rl_script_args.clip_eps,
                                          entropy_coef=config_args.rl_script_args.entropy_coef,
                                          value_loss_coef=config_args.rl_script_args.value_loss_coef,
                                          max_grad_norm=config_args.rl_script_args.max_grad_norm,
                                          ppo_epochs=config_args.rl_script_args.ppo_epochs,
                                          skip_ratio_spikes=config_args.rl_script_args.skip_ratio_spikes,
                                          kl_coef=config_args.rl_script_args.kl_coef,
                                          save_after_update=save_model_and_history,
                                          output_dir=saving_path,
                                          loading_path=loading_path
                                          )

        avg_loss = np.mean([_r['loss'] for _r in update_results])
        avg_policy_loss = np.mean([_r['policy_loss'] for _r in update_results])
        avg_value_loss = np.mean([_r['value_loss'] for _r in update_results])
        history["loss"].append(avg_loss)
        history["policy_loss"].append(avg_policy_loss)
        history["value_loss"].append(avg_value_loss)
        history["actions"].extend([tokenizer.decode(int(_a)) for _a in collected_trajectories['act']])
        history["prompts"].extend(collected_trajectories['obs'])
        print(f"Update loss: {avg_loss}")

        if save_model_and_history:
            # Save history
            with open(f"{saving_path}/history.pkl", "wb") as file:
                pickle.dump(history, file)
            history = reset_history()

    start_epoch = epoch - config_args.rl_script_args.save_freq
    saving_path = f"{config_args.rl_script_args.output_dir}/epochs_{start_epoch}-{epoch}"
    os.makedirs(saving_path, exist_ok=True)
    with open(f"{saving_path}/history.pkl", "wb") as file:
        pickle.dump(history, file)

if __name__ == '__main__':
    main()