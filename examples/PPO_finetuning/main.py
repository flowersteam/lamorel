'''
PPO implementation taken from https://github.com/openai/spinningup
'''

import hydra
from utils.ppo_buffer import PPOBuffer
from utils import scores_to_proba_dists
from utils.generate_prompt import generate_prompt
import torch
import numpy as np

from tqdm import tqdm

import gym
import babyai_text

from lamorel import Caller, lamorel_init
from lamorel import BaseUpdater, BaseModuleFunction

lamorel_init()

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

    def forward(self, forward_outputs, minibatch, tokenized_context, **kwargs):
        if self._model_type == "causal":
            model_head = forward_outputs['hidden_states'][-1][:, len(tokenized_context["input_ids"]) - 1, :]
        else:
            model_head = forward_outputs["decoder_hidden_states"][-1][:, 0, :]

        value = self.value_head_op(model_head.to(self.device))
        return value.cpu()


class PPOUpdater(BaseUpdater):
    def perform_update(self, contexts, candidates, _current_batch_ids, **kwargs):
        if not hasattr(self, 'optimizer'):
            self.optimizer = torch.optim.Adam(self._llm_module.parameters(), kwargs["lr"])

        current_process_buffer = {}
        for k in ['actions', 'advantages', 'returns', 'logprobs']:
            current_process_buffer[k] = kwargs[k][_current_batch_ids]

        for i in tqdm(range(kwargs["ppo_epochs"]), ascii=" " * 9 + ">", ncols=100):
            # Use LLM to compute again action probabilities and value
            output = self._llm_module(['__score', 'value'], contexts=contexts, candidates=candidates, require_grad=True)
            scores = torch.stack([_o['__score'] for _o in output]).squeeze()
            probas = scores_to_proba_dists(scores)
            values = torch.stack([_o["value"][0] for _o in output])

            # Compute policy loss
            entropy = probas.entropy().mean()
            log_prob = probas.log_prob(current_process_buffer['actions'])
            if len(log_prob.shape) > 1:
                log_prob = log_prob.sum(dim=-1)
            ratio = torch.exp(log_prob - current_process_buffer['logprobs'])
            clip_adv = torch.clamp(ratio, 1 - kwargs["clip_eps"], 1 + kwargs["clip_eps"]) * current_process_buffer['advantages']
            policy_loss = -(torch.min(ratio * current_process_buffer['advantages'], clip_adv)).mean()

            # Compute value loss
            value_loss = ((values - current_process_buffer['returns']) ** 2).mean()

            # Compute final loss
            loss = policy_loss - kwargs["entropy_coef"] * entropy + kwargs["value_loss_coef"] * value_loss

            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if kwargs["save_after_update"]:
            print("Saving model...")
            torch.save(self._llm_module.state_dict(), kwargs["output_dir"] + "/model.checkpoint")
            print("Model saved")

        return {'loss': loss}



@hydra.main(config_path='config', config_name='config')
def main(config_args):
    # Random seed
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate environment
    name_env = config_args.rl_script_args.name_environment
    env = gym.make(name_env)
    actions = ["turn_left", "turn_right", "go_forward", "pick_up", "drop", "toggle"]

    # Create LLM agent
    lm_server = Caller(config_args.lamorel_args,
                       custom_updater_class=PPOUpdater,
                       custom_module_functions={
                            'value': ValueHeadModuleFn(config_args.lamorel_args.llm_args.model_type)
                        })

    # Set up experience buffer
    buf = PPOBuffer(config_args.rl_script_args.steps_per_epoch, config_args.rl_script_args.gamma, config_args.rl_script_args.lam)

    # Prepare for interaction with environment
    (_o, infos), ep_ret, ep_len = env.reset(), 0, 0
    o = {
        "mission": _o["mission"],
        "descriptions": infos["descriptions"]
    }

    # Main loop: collect experience in env and update/log each epoch
    n_episodes = 0
    for epoch in range(config_args.rl_script_args.epochs):
        for t in range(config_args.rl_script_args.steps_per_epoch):
            prompt = generate_prompt(o, infos)
            output = lm_server.score(contexts=[prompt], candidates=[actions],
                                     additional_module_function_keys=['value'])[0]
            proba_dist = scores_to_proba_dists(torch.reshape(output['__score'], (1, len(actions))))
            value = output["value"][0]
            action = proba_dist.sample()
            log_prob = proba_dist.log_prob(action)
            a = action.cpu().item()

            _o, r, d, infos = env.step(a)
            next_o = {
                "mission": _o["mission"],
                "descriptions": infos["descriptions"]
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
                if epoch_ended and not (terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.' % ep_len, flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    value = lm_server.custom_module_fns(
                        module_function_keys=['value'],
                        contexts=generate_prompt(o, infos),
                        candidates=actions)[0]["value"][0]
                else:
                    value = 0
                buf.finish_path(value)
                if terminal:
                    n_episodes += 1
                    print(f"Episode {n_episodes}:")
                    print(f"Ret: {ep_ret}")
                    print(f"Len: {ep_len}")
                (o, infos), ep_ret, ep_len = env.reset(), 0, 0

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
                                            lr=config_args.rl_script_args.lr,
                                            clip_eps=config_args.rl_script_args.clip_eps,
                                            entropy_coef=config_args.rl_script_args.entropy_coef,
                                            value_loss_coef=config_args.rl_script_args.value_loss_coef,
                                            ppo_epochs=config_args.rl_script_args.ppo_epochs,
                                            save_after_update=save_model,
                                            output_dir=config_args.rl_script_args.output_dir)
        print(f"Update results: {update_results}")

if __name__ == '__main__':
    main()

