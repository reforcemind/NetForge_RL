from __future__ import annotations
import numpy as np
import pytest
import netforge_rl.environment.parallel_env as parallel_env_module
from netforge_rl.core.functional import apply_state_deltas
from netforge_rl.core.physics import ConflictResolutionEngine
from netforge_rl.environment.parallel_env import NetForgeRLEnv
SCENARIO_CONFIG = {'scenario_type': 'ransomware', 'docker_bridge_mode': 'sim', 'nlp_backend': 'tfidf', 'max_ticks': 30, 'log_latency': 2}

def _scripted_actions(env, rng):
    actions = {}
    for agent in env.agents:
        space = env.action_space(agent)
        actions[agent] = np.array([rng.integers(0, n) for n in space.nvec], dtype=np.int64)
    return actions

class _ResolveSpy:

    def __init__(self):
        self.last_resolved: dict = {}
        self._orig_func = ConflictResolutionEngine.resolve

    def __enter__(self):
        orig = self._orig_func

        @staticmethod
        def spy_resolve(effects):
            resolved = orig(effects)
            self.last_resolved = resolved
            return resolved
        parallel_env_module.ConflictResolutionEngine.resolve = spy_resolve
        return self

    def __exit__(self, *exc):
        parallel_env_module.ConflictResolutionEngine.resolve = staticmethod(self._orig_func)

def _flatten_deltas(resolved_effects) -> dict:
    flat = {}
    for _agent, eff in resolved_effects.items():
        if eff is None or not eff.success:
            continue
        if isinstance(eff.state_deltas, dict):
            for k, v in eff.state_deltas.items():
                flat[k] = v
    return flat

@pytest.mark.integration
def test_pure_apply_matches_legacy_step_host_arrays() -> None:
    env = NetForgeRLEnv(SCENARIO_CONFIG)
    env.reset(seed=42)
    rng = np.random.default_rng(42)
    checked_steps = 0
    with _ResolveSpy() as spy:
        for _ in range(20):
            if not env.agents:
                break
            pre = env.to_envstate()
            env.step(_scripted_actions(env, rng))
            post = env.to_envstate()
            deltas = _flatten_deltas(spy.last_resolved)
            replayed = apply_state_deltas(pre, deltas)
            np.testing.assert_array_equal(replayed.hosts.status, post.hosts.status, err_msg='status array diverged from legacy step')
            np.testing.assert_array_equal(replayed.hosts.privilege, post.hosts.privilege, err_msg='privilege array diverged from legacy step')
            np.testing.assert_array_equal(replayed.hosts.compromised_by_id, post.hosts.compromised_by_id, err_msg='compromised_by_id array diverged from legacy step')
            np.testing.assert_array_equal(replayed.hosts.edr_active, post.hosts.edr_active)
            checked_steps += 1
    assert checked_steps > 0, 'episode terminated immediately; nothing checked'