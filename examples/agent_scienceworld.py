import hydra
from lamorel import Caller, lamorel_init
lamorel_init()

from scienceworld import ScienceWorldEnv
import numpy as np

class LLMAgent:
    def __init__(self, lm_server):
        self._lm_server = lm_server

    def generate_goal(self, initial_obs):
        promt_suffix = "\nThis is an example of what I could do here:"
        prompt = initial_obs + promt_suffix
        result = self._lm_server.generate(contexts=[prompt],
                                          max_new_tokens=15,
                                          do_sample=True,
                                          temperature=1,
                                          top_p=0.70,
                                          top_k=0)

        goal = result[0][0]["text"]
        return goal

    def play(self, obs, possible_actions):
        prompt_suffix = "\nYou choose to "
        scores = self._lm_server.score(contexts=[obs + prompt_suffix], candidates=[list(possible_actions)])
        action_idx = np.argmax(scores[0]).item()
        return possible_actions[action_idx]


# This will be overriden by lamorel's launcher if used
@hydra.main(config_path='config', config_name='config')
def main(config_args):
    # lm server
    lm_server = Caller(config_args.lamorel_args)

    # Env
    env = ScienceWorldEnv('', envStepLimit=10, threadNum=0)

    # Agent
    agent = LLMAgent(lm_server)

    # Play
    task_names = env.getTaskNames()
    env.load(task_names[13], 0, 'easy')
    obs, info = env.resetWithVariation(env.getRandomVariationTest(), 'easy')
    generated_goal = agent.generate_goal(info["look"])
    print(f"Generated goal: {generated_goal}")

    for step in range(1, 10):
        print(f'Step number {step}')
        action = agent.play(info["look"], info["valid"])
        obs, reward, done, info = env.step(action)

    lm_server.close()

if __name__ == '__main__':
    main()
