from __future__ import annotations
import gymnasium
from gymnasium import spaces
import numpy as np
import functools

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from pettingzoo.utils.env_logger import EnvLogger
from pettingzoo.utils.wrappers.base import BaseWrapper
import string


CHARSET = string.ascii_letters + string.digits + string.punctuation + " "

__all__ = {'env', 'raw_env'}

def env(render_mode=None) -> AECEnv:
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = raw_env(render_mode=internal_render_mode)
    # This wrapper translate the environment's observations to text, used for text-based agents
    env = ShampooTextWrapper(env)
    # This wrapper is only for environments which print results to the terminal
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env


class raw_env(AECEnv):
    """
    The metadata holds environment constants. From gymnasium, we inherit the "render_modes",
    metadata which specifies which modes can be put into the render() method.
    At least human mode should be supported.
    The "name" metadata allows the environment to be pretty printed.
    """
    metadata = {"render_modes": ["human"], "name": "Shampoo-v0"}

    def __init__(self, num_shampoos=3, render_mode='human') -> None:

        """
        The init method takes in environment arguments and
         should define the following attributes:
        - possible_agents
        - render_mode

        Note: as of v1.18.1, the action_spaces and observation_spaces attributes are deprecated.
        Spaces should be defined in the action_space() and observation_space() methods.
        If these methods are not overridden, spaces will be inferred from self.observation_spaces/action_spaces, raising a warning.

        These attributes should not be changed after initialization.
        """
        self.possible_agents = ["seller", "buyer"]

        # optional: a mapping between agent name and ID
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

        # Store number of shampoo types
        self.num_shampoos = num_shampoos

        # Create random shampoos for the store
        self.shampoo_properties = np.random.choice([0, 1], (num_shampoos, 3))
        # the maximum value of a shampoo is 3
        self.shampoo_prices = np.random.uniform(1, 3, num_shampoos)
        self.shampoo_values = np.sum(self.shampoo_properties, axis=1)

        # Seller observation space
        self.seller_observation_space = spaces.Dict(
            {
                "properties": spaces.Box(low=0, high=1, shape=(num_shampoos, 3), dtype=int),
                "values": spaces.Box(low=0, high=3, shape=(num_shampoos,), dtype=int),
                "prices": spaces.Box(low=1, high=3, shape=(num_shampoos,), dtype=float)
            }
        )

        # Seller action space
        # self.seller_action_space = spaces.Box(low=0, high=1, shape=(num_shampoos, 3), dtype=int)
        self.seller_action_space = spaces.Text(max_length=512, charset=CHARSET)

        # Buyer observation space
        self.buyer_observation_space = spaces.Dict(
            {
                "information": spaces.Text(max_length=512, charset=CHARSET),
                "prices": spaces.Box(low=1, high=3, shape=(num_shampoos,), dtype=float)
            }
        )

        # Buyer action space
        self.buyer_action_space = spaces.Discrete(num_shampoos)

        # optional: we can define the observation and action spaces here as attributes to be used in their corresponding methods
        self._action_spaces = {"seller": self.seller_action_space, "buyer": self.buyer_action_space}
        self._observation_spaces = {"seller": self.seller_observation_space, "buyer": self.buyer_observation_space}
        self.render_mode = render_mode

    # Observation space should be defined here.
    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent) -> spaces.Space:
        return self._observation_spaces[agent]
    
    # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent) -> spaces.Space:
        return self._action_spaces[agent]
    
    def render(self) -> None:
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return
        elif self.render_mode == "human":
            if self.state[self.agent_selection] is not None:
                if self.agent_selection == "seller" and "seller" in self.terminations.keys() and (not self.terminations["seller"]):
                    print("Shampoo Store:")
                    for i in range(self.num_shampoos):
                        print(f"Shampoo {i + 1}:")
                        print(f"\tProperties: {self.shampoo_properties[i]} (Cleanliness, Hair Protection, Safety)")
                        print(f"\tValue: {self.shampoo_values[i]}")
                        print(f"\tPrice: ${self.shampoo_prices[i]:.2f}")
                    print("--------")
                    seller_desc = self.state[self.agent_selection]
                    print("Seller's Descriptions:", seller_desc)
                    print("--------")
                elif self.agent_selection == "buyer" and "buyer" in self.terminations.keys() and (not self.terminations["buyer"]):
                    print("Shampoo Store:")
                    for i in range(self.num_shampoos):
                        print(f"Shampoo {i + 1}:")
                        print(f"\tProperties: {self.shampoo_properties[i]} (Cleanliness, Hair Protection, Safety)")
                        print(f"\tValue: {self.shampoo_values[i]}")
                        print(f"\tPrice: ${self.shampoo_prices[i]:.2f}")
                    print("--------")
                    seller_desc = self.state[self.agents[0]]
                    print("Seller's Descriptions:", seller_desc)
                    print("--------")
                    buyer_decision = self.state[self.agent_selection]
                    print("Buyer's Decisions:")
                    print(f"Shampoo {buyer_decision + 1} is chosen by the buyer.")
                    print("--------")

    def observe(self, agent) -> dict:
        """
        Observe should return the observation of the specified agent. This function
        should return a sane observation (though not necessarily the most up to date possible)
        at any time after reset() is called.
        """
        if agent == "seller":
            return self.get_seller_observation()
        elif agent == "buyer":
            # assert self._agent_selector.is_last(), "Buyer can only observe after seller has given a description."
            assert self.state["seller"] is not None, "Buyer can only observe after seller has given a description."
            buyer_observation = {
                "information": self.state["seller"],
                "prices": self.shampoo_prices,
            }
            return buyer_observation

    def close(self) -> None:
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        pass

    def reset(self, seed=None, options=None) -> None:
        """
        Reset needs to initialize the following attributes
        - agents
        - rewards
        - _cumulative_rewards
        - terminations
        - truncations
        - infos
        - agent_selection
        And must set up the environment so that render(), step(), and observe()
        can be called without issues.
        Here it sets up the state dictionary which is used by step() and the observations dictionary which is used by step() and observe()
        """
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = {agent: None for agent in self.agents}
        self.observations = {agent: None for agent in self.agents}
        # self.num_moves = 0
        """
        Our agent_selector utility allows easy cyclic stepping through the agents list.
        """
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

    def get_seller_observation(self) -> dict:
        """
        Return the seller observation.
        """
        return {
            "properties": self.shampoo_properties,
            "values": self.shampoo_values,
            "prices": self.shampoo_prices,
        }
    
    def step(self, action) -> None:
        """
        step(action) takes in an action for the current agent (specified by
        agent_selection) and needs to update
        - rewards
        - _cumulative_rewards (accumulating the rewards)
        - terminations
        - truncations
        - infos
        - agent_selection (to the next agent)
        And any internal state used by observe() or render()
        """
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            # handles stepping an agent which is already dead
            # accepts a None action for the one agent, and moves the agent_selection to
            # the next dead agent,  or if there are no more dead agents, to the next live agent
            self._was_dead_step(action)
            return
        
        agent = self.agent_selection

        # the agent which stepped last had its _cumulative_rewards accounted for
        # (because it was returned by last()), so the _cumulative_rewards for this
        # agent should start again at 0
        self._cumulative_rewards[agent] = 0

        # stores action of the agent which stepped
        self.state[self.agent_selection] = action
        
        # collect reward if it is the last agent to act
        if self._agent_selector.is_last():
            # Seller reward
            sold_shampoo_price = self.shampoo_prices[action]
            sold_shampoo_value = self.shampoo_values[action]
            self.rewards[self.agents[0]] = sold_shampoo_price - sold_shampoo_value
            # Buyer reward
            self.rewards[self.agents[1]] = sold_shampoo_value - sold_shampoo_price
        else:
            # necessary so that observe() returns a reasonable observation at all times.
            self.state[self.agents[1 - self.agent_name_mapping[agent]]] = None
            # no rewards are allocated until both players give an action
            self._clear_rewards()

        if self.render_mode == "human":
            self.render()

        # update terminations
        if self._agent_selector.is_last():
            self.terminations = {agent: True for agent in self.agents}

        # selects the next agent.
        self.agent_selection = self._agent_selector.next()
        # Adds .rewards to ._cumulative_rewards
        self._accumulate_rewards()


class ShampooTextWrapper(BaseWrapper):
    """
    This wrapper converts the outputs of observe() to text.
    """
    def __init__(self, env:AECEnv) -> None:
        """
        Initialize the wrapper.
        """
        super().__init__(env)
        assert isinstance(
            env, AECEnv
        ), "ShampooTextWrapper is only compatible with AEC environments."

        self.seller_observation_space = spaces.Text(max_length=512, charset=CHARSET)
        self.buyer_observation_space = spaces.Text(max_length=512, charset=CHARSET)
        self._observation_spaces = {"seller": self.seller_observation_space, "buyer": self.buyer_observation_space}

    # Observation space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent) -> spaces.Space:
        """
        Returns the observation space of the specified agent.
        ------------------------------------------------
        Input:
            agent: the specified agent
        Output:
            obs_space: the observation space of the specified agent
        ------------------------------------------------
        """
        return self._observation_spaces[agent]

    def observe(self, agent) -> str:
        """
        Returns the observation of the specified agent in a string format.
        ------------------------------------------------
        Input:
            agent: the specified agent
        Output:
            obs_str: the observation of the specified agent in a string format
        ------------------------------------------------
        """
        obs = super().observe(agent)
        if agent == "seller":
            return self._format_seller_observation(obs)
        elif agent == "buyer":
            return self._format_buyer_observation(obs)

    def _format_seller_observation(self, obs) -> str:
        """
        Format the seller observation to a string.
        ------------------------------------------------
        Input:
            obs: the seller observation, not used in this function
        Output:
            obs_str: the formatted seller observation
        ------------------------------------------------
        """
        obs_str = ''
        for i in range(self.num_shampoos):
            obs_str += f"Shampoo {i + 1}:" + "\n"
            obs_str += f"\tProperties: {self.shampoo_properties[i]} (Cleanliness, Hair Protection, Safety)" + "\n"
            obs_str += f"\tValue: {self.shampoo_values[i]}" + "\n"
            obs_str += f"\tPrice: ${self.shampoo_prices[i]:.2f}" + "\n"
        return obs_str
    
    def _format_buyer_observation(self, obs) -> str:
        """
        Format the buyer observation to a string.
        ------------------------------------------------
        Input:
            obs: the buyer observation
        Output:
            obs_str: the formatted buyer observation
        ------------------------------------------------
        """
        obs_str = "The seller's descriptions:" + "\n"
        obs_str += obs["information"] + "\n"
        obs_str += "The shampoo prices:" + "\n"
        for i in range(self.num_shampoos):
            obs_str += f"Shampoo {i + 1}: ${self.shampoo_prices[i]:.2f}" + "\n"
        return obs_str

    def __str__(self) -> str:
        """
        Return the name of the environment.
        """
        return str(self.env)