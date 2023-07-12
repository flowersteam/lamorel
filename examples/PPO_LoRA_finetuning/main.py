'''
PPO implementation taken from https://github.com/openai/spinningup
'''

import hydra
from utils.ppo_buffer import PPOBuffer
from utils.generate_prompt import generate_prompt
import torch
import numpy as np

from peft import LoraConfig, get_peft_model

from tqdm import tqdm
import time
import pickle
import math
import os

import gym
import babyai_text

from lamorel import Caller, lamorel_init
from lamorel import BaseUpdater, BaseModuleFunction, BaseModelInitializer

lamorel_init()

class LogScoringModuleFn(BaseModuleFunction):
    def __init__(self, model_type):
        super().__init__()
        self._model_type = model_type
        self._pad_token = 0

    def initialize(self):
        pass

    def forward(self, forward_outputs, minibatch, tokenized_contexts, **kwargs):
        if self._model_type == "causal":  # hence input should be removed from result
            raise NotImplementedError()
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
        masked_token_probs = tokens_logprobs.masked_fill(mask, 1.0)  # apply mask
        minibatch_probs = masked_token_probs.sum(-1)  # compute final sequences' probability

        return minibatch_probs.cpu()

class ValueHeadModuleFn(BaseModuleFunction):
    def __init__(self, model_type):
        super().__init__()
        self._model_type = model_type

    def initialize(self):
        llm_hidden_size = self.llm_config.to_dict()[self.llm_config.attribute_map['hidden_size']]
        self.value_head_op = torch.nn.Sequential(
            torch.nn.Linear(llm_hidden_size, 1024),
            torch.nn.Sigmoid(),
            torch.nn.Linear(1024, 1024),
            torch.nn.Sigmoid(),
            torch.nn.Linear(1024, 1),
        ).to(self.device)

    def forward(self, forward_outputs, minibatch, tokenized_contexts, **kwargs):
        if self._model_type == "causal":
            model_head = forward_outputs['hidden_states'][-1][:, len(tokenized_contexts["input_ids"]) - 1, :]
        else:
            model_head = forward_outputs["decoder_hidden_states"][-1][:, 0, :]

        value = self.value_head_op(model_head.to(torch.float32).to(self.device))
        return value.cpu()

class PeftInitializer(BaseModelInitializer):
    def __init__(self, model_type, use_lora, use_fp16):
        super().__init__()
        self._model_type = model_type
        self._use_lora = use_lora
        self._use_fp16 = use_fp16

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

    def initialize_model(self, model):
        if self._use_lora:
            llm_module = model._modules['_LLM_model']

            if self._use_fp16:
                llm_module.half()

            llm_module.gradient_checkpointing_enable()  # reduce number of stored activations
            llm_module.enable_input_require_grads()

            # Init adapters
            config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q", "v"] if self._model_type == "seq2seq" else ["q_proj", "v_proj"],
                lora_dropout=0.0,
                bias="none",
                task_type="SEQ_2_SEQ_LM" if self._model_type == "seq2seq" else "CAUSAL_LM"
            )
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
        self._print_trainable_parameters(model)
        return model

class PPOLoRAUpdater(BaseUpdater):
    def __init__(self, model_type, minibatch_size, gradient_batch_size):
        super(PPOLoRAUpdater, self).__init__()
        self._model_type = model_type
        self._minibatch_size = minibatch_size
        self._gradient_batch_size = gradient_batch_size

    def _get_memory_allocated(self):
        return math.floor(torch.cuda.memory_allocated() / 1024 / 1024)

    def perform_update(self, contexts, candidates, _current_batch_ids, **kwargs):
        times = {
            "forward": [],
            "backward": [],
            "optim": []
        }

        gpu_usage = {
            "forward": [],
            "backward": [],
            "optim": []
        }

        iterator_trainable_params = self._llm_module.parameters()
        if not hasattr(self, 'optimizer'):
            self.optimizer = torch.optim.Adam(iterator_trainable_params, kwargs["lr"])
            gpu_usage["init_optim"] = self._get_memory_allocated()

        current_process_buffer = {}
        for k in ['actions', 'advantages', 'returns', 'logprobs', 'values']:
            current_process_buffer[k] = kwargs[k][_current_batch_ids]

        longest_candidate = max([len(_c) for _c in candidates])
        _batch_size = self._gradient_batch_size
        _full_time = time.time()
        for i in tqdm(range(kwargs["ppo_epochs"]), ascii=" " * 9 + ">", ncols=100):
            for step in range(len(contexts) // self._minibatch_size + 1):
                if step * self._minibatch_size >= len(contexts):
                    break
                self.optimizer.zero_grad()
                for accumulated_batch in range(self._minibatch_size // self._gradient_batch_size + 1):
                    _start_idx = step * self._minibatch_size + accumulated_batch * self._gradient_batch_size
                    _stop_idx = step * self._minibatch_size + \
                                min(
                                    (accumulated_batch + 1) * self._gradient_batch_size,
                                    (step + 1) * self._minibatch_size)

                    _contexts = contexts[_start_idx:_stop_idx]
                    _candidates = candidates[_start_idx:_stop_idx]
                    if len(_contexts) == 0: break
                    # Use LLM to compute again action probabilities and value
                    _time = time.time()
                    output = self._llm_module(['score', 'value'], contexts=_contexts, candidates=_candidates,
                                              require_grad=True, minibatch_size=_batch_size * longest_candidate)
                    times["forward"].append(time.time() - _time)
                    gpu_usage["forward"].append(self._get_memory_allocated())
                    scores = torch.stack([_o['score'] for _o in output]).squeeze()
                    probas = torch.distributions.Categorical(logits=scores)
                    values = torch.stack([_o["value"][0] for _o in output]).squeeze()

                    # Compute policy loss
                    entropy = probas.entropy().mean()
                    log_prob = probas.log_prob(current_process_buffer['actions'][_start_idx:_stop_idx]) # Use logprobs from dist as they were normalized
                    ratio = torch.exp(log_prob - current_process_buffer['logprobs'][_start_idx:_stop_idx])
                    clip_adv = torch.clamp(ratio, 1 - kwargs["clip_eps"], 1 + kwargs["clip_eps"]) * current_process_buffer['advantages'][_start_idx:_stop_idx]
                    policy_loss = -(torch.min(ratio * current_process_buffer['advantages'][_start_idx:_stop_idx], clip_adv)).mean()

                    # Compute value loss
                    unclipped_value_error = ((values - current_process_buffer['returns'][_start_idx:_stop_idx]) ** 2)
                    clipped_values = current_process_buffer['values'][_start_idx:_stop_idx] + \
                                     torch.clamp(values - current_process_buffer['values'][_start_idx:_stop_idx],
                                                 -kwargs["clip_eps"], kwargs["clip_eps"])
                    clipped_value_error = ((clipped_values - current_process_buffer['returns'][_start_idx:_stop_idx]) ** 2)
                    value_loss = torch.max(unclipped_value_error, clipped_value_error).mean()

                    # Compute final loss
                    loss = policy_loss - kwargs["entropy_coef"] * entropy + kwargs["value_loss_coef"] * value_loss

                    # Backward
                    _time = time.time()
                    loss.backward()
                    times["backward"].append(time.time() - _time)
                    gpu_usage["backward"].append(self._get_memory_allocated())
                _time = time.time()
                torch.nn.utils.clip_grad_norm_(iterator_trainable_params, kwargs["max_grad_norm"])
                self.optimizer.step()
                times["optim"].append(time.time() - _time)
                gpu_usage["optim"].append(self._get_memory_allocated())

        times["full_update"] = time.time() - _full_time

        if kwargs["save_after_update"]:
            print("Saving model...")
            torch.save(self._llm_module.state_dict(), kwargs["output_dir"] + "/model.checkpoint")
            print("Model saved")

        return {'loss': loss, 'value_loss': value_loss, 'policy_loss': policy_loss, 'times': times, 'gpu_usage': gpu_usage}



@hydra.main(config_path='config', config_name='config')
def main(config_args):
    # Random seed
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.makedirs(config_args.rl_script_args.output_dir, exist_ok=True)

    # Instantiate environment
    name_env = config_args.rl_script_args.name_environment
    env = gym.make(name_env)
    actions = ["turn left", "turn right", "go forward"] #"pick_up", "drop", "toggle"]

    # Create LLM agent
    lm_server = Caller(config_args.lamorel_args,
                       custom_updater=PPOLoRAUpdater(config_args.lamorel_args.llm_args.model_type,
                                                     config_args.rl_script_args.minibatch_size,
                                                     config_args.rl_script_args.gradient_batch_size),
                       custom_model_initializer=PeftInitializer(config_args.lamorel_args.llm_args.model_type,
                                                                config_args.rl_script_args.use_lora,
                                                                config_args.rl_script_args.use_fp16),
                       custom_module_functions={
                            'score': LogScoringModuleFn(config_args.lamorel_args.llm_args.model_type),
                            'value': ValueHeadModuleFn(config_args.lamorel_args.llm_args.model_type)
                        })

    # Set up experience buffer
    buf = PPOBuffer(config_args.rl_script_args.steps_per_epoch, config_args.rl_script_args.gamma, config_args.rl_script_args.lam)

    # Prepare for interaction with environment
    (_o, _infos), ep_ret, ep_len = env.reset(), 0, 0
    o = {
        "mission": _o["mission"],
        "descriptions": _infos["descriptions"]
    }

    # Main loop: collect experience in env and update/log each epoch
    n_episodes = 0
    _time = time.time()
    history = {
        "ep_len": [],
        "ep_ret": [],
        "loss": [],
        "policy_loss": [],
        "value_loss": [],
        "actions": [],
        "prompts": [],
        "training_times": [],
        "collection_times": [],
        "training_gpu_usage": []
    }

    for epoch in range(config_args.rl_script_args.epochs):
        __time = time.time()
        for t in range(config_args.rl_script_args.steps_per_epoch):
            prompt = generate_prompt(o)
            output = lm_server.custom_module_fns(['score', 'value'],
                                                 contexts=[prompt],
                                                 candidates=[actions],
                                                 minibatch_size=len(actions))[0]
            _scores = torch.reshape(output['score'], (1, len(actions)))
            proba_dist = torch.distributions.Categorical(logits=_scores)
            value = output["value"][0]
            action = proba_dist.sample()
            log_prob = proba_dist.log_prob(action) #torch.index_select(_scores, 1, action) #proba_dist.log_prob(action)
            a = action.cpu().item()

            _o, r, d, _infos = env.step(a)
            next_o = {
                "mission": _o["mission"],
                "descriptions": _infos["descriptions"]
            }
            ep_ret += r
            ep_len += 1

            # save and log
            buf.store(prompt, a, r, value, log_prob)

            # Update obs (critical!)
            o = next_o

            timeout = ep_len == config_args.rl_script_args.max_ep_len
            terminal = d or timeout
            epoch_ended = t == config_args.rl_script_args.steps_per_epoch - 1

            if terminal or epoch_ended:
                if not terminal:
                    print('Warning: trajectory cut off by epoch at %d steps.' % ep_len, flush=True)
                    # if trajectory didn't reach terminal state, bootstrap value target
                    value = lm_server.custom_module_fns(
                        module_function_keys=['value'],
                        contexts=[generate_prompt(o)],
                        candidates=[actions[0]],
                        minibatch_size=1
                    )[0]["value"][0]
                else:
                    value = 0

                buf.finish_path(value)
                if terminal:
                    n_episodes += 1
                    print(f"Episode {n_episodes}:")
                    print(f"Ret: {ep_ret}")
                    print(f"Len: {ep_len}")
                    history["ep_len"].append(ep_len)
                    history["ep_ret"].append(ep_ret)
                    (_o, _infos), ep_ret, ep_len = env.reset(), 0, 0
                    o = {
                        "mission": _o["mission"],
                        "descriptions": _infos["descriptions"]
                    }

        history["collection_times"].append(time.time() - __time)
        # Perform PPO update!
        print(f"PPO update number {epoch + 1}")
        save_model = (epoch % config_args.rl_script_args.save_freq == 0 or
                      epoch == config_args.rl_script_args.epochs - 1) and epoch != 0
        collected_trajectories = buf.get()
        update_results = lm_server.update(collected_trajectories['obs'],
                                            [actions for _ in range(config_args.rl_script_args.steps_per_epoch)],
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
                                            save_after_update=save_model,
                                            output_dir=config_args.rl_script_args.output_dir)
        avg_loss = np.mean([_r['loss'].detach().cpu().item() for _r in update_results])
        avg_policy_loss = np.mean([_r['policy_loss'].detach().cpu().item() for _r in update_results])
        avg_value_loss = np.mean([_r['value_loss'].detach().cpu().item() for _r in update_results])
        history["loss"].append(avg_loss)
        history["policy_loss"].append(avg_policy_loss)
        history["value_loss"].append(avg_value_loss)
        history["actions"].append([actions[int(_a.item())] for _a in collected_trajectories['act']])
        history["prompts"].append(collected_trajectories['obs'])
        history["training_times"].append([r["times"] for r in update_results])
        history["training_gpu_usage"].append([r["gpu_usage"] for r in update_results])
        print(f"Update loss: {avg_loss}")
        with open(config_args.rl_script_args.output_dir + "/history.pkl", "wb") as file:
            pickle.dump(history, file)

    print(f"Training took {time.time() - _time} seconds")
    with open(config_args.rl_script_args.output_dir + "/history.pkl", "wb") as file:
        pickle.dump(history, file)

if __name__ == '__main__':
    main()