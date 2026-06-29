from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip('jax')
jnp = pytest.importorskip('jax.numpy')

from netforge_rl.backends.jax import (
    SCENARIO_IDS,
    BatchedActions,
    VectorEnvSpec,
    initial_batched_state,
    make_vector_step,
    random_actions,
    scenario_done,
    to_jax,
)
from netforge_rl.backends.reference import reference_done, reference_step
from netforge_rl.core.functional import from_global_state
from netforge_rl.topologies.network_generator import NetworkGenerator

AGENTS = ('red_operator', 'blue_dmz', 'blue_internal', 'blue_restricted')
N_STEPS = 25
N_SEEDS = 4


def _np_template(seed: int):
    legacy = NetworkGenerator().generate(seed=seed)
    snap = from_global_state(legacy, agent_ids=AGENTS)
    jstate = to_jax(snap)
    return jax.tree_util.tree_map(lambda x: np.asarray(x), jstate)


def _slice_actions(actions: BatchedActions, b: int) -> BatchedActions:
    return BatchedActions(
        red_target_idx=np.asarray(actions.red_target_idx[b]),
        blue_target_idx=np.asarray(actions.blue_target_idx[b]),
        red_attempt=np.asarray(actions.red_attempt[b]),
        blue_attempt=np.asarray(actions.blue_attempt[b]),
        red_action_type=np.asarray(actions.red_action_type[b]),
        blue_action_type=np.asarray(actions.blue_action_type[b]),
    )


@pytest.mark.integration
@pytest.mark.parametrize('scenario_name', list(SCENARIO_IDS.keys()))
def test_jax_kernel_matches_numpy_reference(scenario_name) -> None:
    scenario = SCENARIO_IDS[scenario_name]
    spec = VectorEnvSpec(n_hosts=100, n_red=1, n_blue=3, scenario=scenario)
    step = make_vector_step(spec)

    np_template = _np_template(seed=7)
    jax_state = initial_batched_state(
        to_jax(
            from_global_state(NetworkGenerator().generate(seed=7), agent_ids=AGENTS)
        ),
        batch_size=N_SEEDS,
    )
    ref_states = [
        jax.tree_util.tree_map(lambda x: np.array(x), np_template)
        for _ in range(N_SEEDS)
    ]

    key = jax.random.PRNGKey(0)
    for t in range(N_STEPS):
        key, sub = jax.random.split(key)
        actions = random_actions(spec, batch_size=N_SEEDS, key=sub)

        jax_state, jax_rewards = step(jax_state, actions)
        jax_rewards = np.asarray(jax_rewards)
        jax_done = np.asarray(scenario_done(jax_state, spec))

        for b in range(N_SEEDS):
            ref_states[b], ref_reward = reference_step(
                ref_states[b], _slice_actions(actions, b), spec
            )
            ref_done = reference_done(ref_states[b], spec)

            jh = jax_state.hosts
            rh = ref_states[b].hosts
            for field in (
                'status',
                'privilege',
                'compromised_by_id',
                'system_integrity',
                'decoy',
                'edr_active',
                'contains_honeytokens',
            ):
                np.testing.assert_array_equal(
                    np.asarray(getattr(jh, field))[b],
                    np.asarray(getattr(rh, field)),
                    err_msg=f'{scenario_name} step {t} env {b}: {field} diverged',
                )
            np.testing.assert_allclose(
                jax_rewards[b],
                ref_reward,
                rtol=1e-5,
                atol=1e-5,
                err_msg=f'{scenario_name} step {t} env {b}: reward diverged',
            )
            assert bool(jax_done[b]) == bool(ref_done), (
                f'{scenario_name} step {t} env {b}: done diverged'
            )
