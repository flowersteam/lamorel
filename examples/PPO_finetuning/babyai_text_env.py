import gym
import babyai_text
import babyai.utils as utils
from babyai.paral_env_simple import ParallelEnv

class BabyAITextEnv:
    def __init__(self, config_dict):
        self.n_parallel = config_dict["number_envs"]
        self._action_space = [a.replace("_", " ") for a in config_dict["action_space"]]
        envs = []
        for i in range(config_dict["number_envs"]):
            env = gym.make(config_dict["task"])
            env.seed(100 * config_dict["seed"] + i)
            envs.append(env)

        self._env = ParallelEnv(envs)

    def __prepare_infos(self, obs, infos):
        for _obs, info in zip(obs, infos):
            info["possible_actions"] = self._action_space
            info["goal"] = f"Goal of the agent: {_obs['mission']}"
        return list(infos)

    def __generate_obs(self, obs, infos):
        return [info["descriptions"] for info in infos]
    def reset(self):
        obs, infos = self._env.reset()
        return self.__generate_obs(obs, infos), self.__prepare_infos(obs, infos)
    def step(self, actions_id, actions_command):
        obs, rews, dones, infos = self._env.step(actions_id)
        return self.__generate_obs(obs, infos), \
                [rew * 20.0 for rew in rews], \
                dones, \
                self.__prepare_infos(obs, infos)