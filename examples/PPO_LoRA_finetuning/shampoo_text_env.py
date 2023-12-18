import gym
import babyai_text
import babyai.utils as utils
# from babyai.paral_env_simple import ParallelEnv
from shampoo import shampoo_v0
from shampoo.shampoo_v0 import ParallelEnv
import pprint

class ShampooTextEnv:
    def __init__(self, config_dict):
        self.n_parallel = config_dict["number_envs"]
        # self._action_space = [a.replace("_", " ") for a in config_dict["action_space"]]
        envs = []
        for i in range(config_dict["number_envs"]):
            # env = gym.make(config_dict["task"])
            env = shampoo_v0.env()
            env.reset(100 * config_dict["seed"] + i)
            envs.append(env)

        self._env = ParallelEnv(envs)

    def __prepare_infos(self, obs, infos):
        for _obs, info in zip(obs, infos):
            # info["possible_actions"] = self._action_space
            # info["goal"] = f"Goal of the agent: {_obs['mission']}"
            for key in info.keys():
                if "prompt_prefix" in info[key].keys() and "prompt_suffix" in info[key].keys():
                    info[key]["prompt"] = "".join(
                        [info[key]["prompt_prefix"], 
                        _obs[key], 
                        info[key]["prompt_suffix"]])
                    del info[key]["prompt_prefix"]
                    del info[key]["prompt_suffix"]
        return list(infos)

    def __generate_obs(self, obs, infos):
        # return [info["descriptions"] for info in infos]
        return obs
    
    def reset(self, seeds):
        obs, infos = self._env.reset(seeds)
        return self.__generate_obs(obs, infos), self.__prepare_infos(obs, infos)
    
    def step(self, actions):
        obs, rews, dones, infos = self._env.step(actions)
        # return self.__generate_obs(obs, infos), \
        #         [rew * 20.0 for rew in rews], \
        #         dones, \
        #         self.__prepare_infos(obs, infos)
        return self.__generate_obs(obs, infos), \
                rews, \
                dones, \
                self.__prepare_infos(obs, infos)
    

if __name__ == '__main__':
    config_dict = {
        "number_envs": 4,
        "seed": 0,
    }
    env = ShampooTextEnv(config_dict)
    seeds = [0] * config_dict["number_envs"]
    obs, infos = env.reset(seeds)
    # pretty print
    pprint = pprint.PrettyPrinter(indent=4).pprint
    pprint(obs)
    pprint(infos)
    actions = []
    for i in range(config_dict["number_envs"]):
        actions.append({'seller': '0'})
    obs, rews, dones, infos = env.step(actions)
    pprint(obs)
    pprint(rews)
    pprint(dones)
    pprint(infos)
    actions = []
    for i in range(config_dict["number_envs"]):
        actions.append({'buyer': 1})
    obs, rews, dones, infos = env.step(actions)
    pprint(obs)
    pprint(rews)
    pprint(dones)
    pprint(infos)