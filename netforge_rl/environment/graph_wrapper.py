from pettingzoo.utils.wrappers import BaseParallelWrapper

from netforge_rl.core.graph_obs import build_graph_observation


class GraphObservationWrapper(BaseParallelWrapper):
    """Injects a graph view into each agent's info['graph'] without changing the obs
    dict. Red agents get a fog-of-war graph; blue agents see all hosts."""

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
        for agent in infos:
            infos[agent]['graph'] = build_graph_observation(gs, agent_id=agent)
        return infos
