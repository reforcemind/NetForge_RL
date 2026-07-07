import hashlib

import numpy as np
import pytest

from netforge_rl.environment.parallel_env import NetForgeRLEnv

SCENARIOS = ['ransomware', 'apt_espionage', 'ot_stuxnet']


def _scripted_actions(env, rng):
    return {
        agent: np.array(
            [rng.integers(0, n) for n in env.action_space(agent).nvec], dtype=np.int64
        )
        for agent in env.agents
    }


def _replay_digest(seed, scenario='ransomware', max_ticks=30):
    """Hash the full observable stream: observations, SIEM embeddings, infos, rewards.

    A stable digest under a fixed seed is the reproducibility guarantee the benchmark
    claims. The golden-trajectory test only covers rewards/terminations; this covers
    everything a learning agent actually sees.
    """
    env = NetForgeRLEnv({'scenario_type': scenario, 'max_ticks': max_ticks})
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    h = hashlib.sha256()
    while env.agents:
        obs, rewards, term, trunc, infos = env.step(_scripted_actions(env, rng))
        for agent in sorted(obs):
            for key in sorted(obs[agent]):
                h.update(np.ascontiguousarray(obs[agent][key]).tobytes())
            h.update(np.float64(rewards[agent]).tobytes())
            for k in sorted(infos.get(agent, {})):
                v = infos[agent][k]
                if isinstance(v, (int, float)):
                    h.update(np.float64(v).tobytes())
        if all(term.values()) or all(trunc.values()):
            break
    return h.hexdigest()


@pytest.mark.fast
@pytest.mark.parametrize('scenario', SCENARIOS)
def test_observation_stream_is_reproducible_under_seed(scenario):
    """Same seed must reproduce observations, SIEM embeddings, infos, and rewards bit-for-bit."""
    a = _replay_digest(seed=123, scenario=scenario)
    b = _replay_digest(seed=123, scenario=scenario)
    assert a == b, f'{scenario}: observation stream not reproducible under a fixed seed'


@pytest.mark.fast
def test_different_seeds_diverge():
    """Distinct seeds must produce distinct observation streams (no accidental constant)."""
    assert _replay_digest(seed=1) != _replay_digest(seed=2)


@pytest.mark.fast
def test_interleaved_envs_stay_independent():
    """Two envs stepped alternately in one process must each reproduce their solo run.

    Guards against shared module-global RNG in the SIEM templates leaking state
    between concurrent environments (the parallel-reproducibility fix)."""
    solo_a = _replay_digest(seed=11, max_ticks=20)
    solo_b = _replay_digest(seed=22, max_ticks=20)

    env_a = NetForgeRLEnv({'scenario_type': 'ransomware', 'max_ticks': 20})
    env_b = NetForgeRLEnv({'scenario_type': 'ransomware', 'max_ticks': 20})
    env_a.reset(seed=11)
    env_b.reset(seed=22)
    rng_a = np.random.default_rng(11)
    rng_b = np.random.default_rng(22)

    def _digest(env, rng, other, other_rng):
        h = hashlib.sha256()
        while env.agents:
            obs, rewards, term, trunc, infos = env.step(_scripted_actions(env, rng))
            for agent in sorted(obs):
                for key in sorted(obs[agent]):
                    h.update(np.ascontiguousarray(obs[agent][key]).tobytes())
                h.update(np.float64(rewards[agent]).tobytes())
                for k in sorted(infos.get(agent, {})):
                    v = infos[agent][k]
                    if isinstance(v, (int, float)):
                        h.update(np.float64(v).tobytes())
            if other.agents:  # interleave a step of the other env
                other.step(_scripted_actions(other, other_rng))
            if all(term.values()) or all(trunc.values()):
                break
        return h.hexdigest()

    inter_a = _digest(env_a, rng_a, env_b, rng_b)
    assert inter_a == solo_a, 'env A perturbed by a concurrent env sharing template RNG'
    # sanity: env B kept running independently and was itself reproducible
    assert solo_b == _replay_digest(seed=22, max_ticks=20)
