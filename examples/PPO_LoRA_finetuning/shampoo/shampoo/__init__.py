# from gym.envs.registration import register

# register(
#     id='Shampoo-v0',
#     entry_point='shampoo.envs:ShampooEnv',
#     max_episode_steps=1,
# )

from pettingzoo.utils.deprecated_module import deprecated_handler


def __getattr__(env_name):
    return deprecated_handler(env_name, __path__, __name__)