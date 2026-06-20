import gymnasium as gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from netforge_rl.environment.parallel_env import NetForgeRLEnv


class NetForgeRLlibEnv(MultiAgentEnv):
    def __init__(self, config=None):
        super().__init__()
        config = config or {}
        self._env = NetForgeRLEnv(config)
        self.possible_agents = self._env.possible_agents
        self.agents = self._env.agents

        self.observation_space = gym.spaces.Dict(
            {
                agent: self._env.observation_space(agent)
                for agent in self.possible_agents
            }
        )
        self.action_space = gym.spaces.Dict(
            {agent: self._env.action_space(agent) for agent in self.possible_agents}
        )

        self._agent_ids = set(self.possible_agents)
        self._obs_space_in_preferred_format = True
        self._action_space_in_preferred_format = True

    def reset(self, *, seed=None, options=None):
        obs_dict, info_dict = self._env.reset(seed=seed, options=options)
        return obs_dict, info_dict

    def step(self, action_dict):
        obs_dict, rew_dict, term_dict, trunc_dict, info_dict = self._env.step(
            action_dict
        )

        term_dict['__all__'] = all(term_dict.values()) if term_dict else True
        trunc_dict['__all__'] = all(trunc_dict.values()) if trunc_dict else True

        return obs_dict, rew_dict, term_dict, trunc_dict, info_dict

    @property
    def observation_spaces(self):
        return self._env.observation_spaces

    @property
    def action_spaces(self):
        return self._env.action_spaces
