from pettingzoo.utils.wrappers import BaseParallelWrapper

from netforge_rl.core.graph_obs import build_graph_observation, build_oracle_graph


class GraphObservationWrapper(BaseParallelWrapper):
    """``info['graph']``: SIEM belief (Blue) / fog (Red). Oracle only if ``oracle=True``."""

    def __init__(self, env, oracle: bool = False):
        super().__init__(env)
        self.oracle = oracle

    def reset(self, seed=None, options=None):
        obs, infos = super().reset(seed=seed, options=options)
        return obs, self._inject(infos)

    def step(self, actions):
        obs, rewards, terminations, truncations, infos = super().step(actions)
        return obs, rewards, terminations, truncations, self._inject(infos)

    def _inject(self, infos):
        gs = getattr(self.env.unwrapped, 'global_state', None)
        if gs is None:
            return infos
        oracle_graph = build_oracle_graph(gs) if self.oracle else None
        for agent in infos:
            infos[agent]['graph'] = build_graph_observation(gs, agent_id=agent)
            if oracle_graph is not None:
                infos[agent]['oracle_graph'] = oracle_graph
        return infos
