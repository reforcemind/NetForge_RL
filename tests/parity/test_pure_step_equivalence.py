"""Episode-level equivalence: pure ``apply_state_deltas`` reproduces the
legacy step's host-array mutations.

The claim we lock in: for every step, given the pre-step :class:`EnvState`
snapshot and the dict of resolved ``ActionEffect.state_deltas`` produced by
the legacy engine, applying those deltas through the functional interpreter
produces a state whose vectorized host arrays equal the post-step snapshot.

This is the load-bearing parity invariant for Phase 2 — the JAX backend
will replace the legacy step loop with a vmap'd functional step. If this
test stays green throughout slices 4/5, the JAX backend can be wired in
with high confidence that its physics matches.

We capture resolved effects by monkey-patching
:meth:`ConflictResolutionEngine.resolve` since adding instrumentation hooks
to production code paths just for tests would be ugly.
"""

from __future__ import annotations

import numpy as np
import pytest

import netforge_rl.environment.parallel_env as parallel_env_module
from netforge_rl.core.functional import apply_state_deltas
from netforge_rl.core.physics import ConflictResolutionEngine
from netforge_rl.environment.parallel_env import NetForgeRLEnv


SCENARIO_CONFIG = {
    'scenario_type': 'ransomware',
    'sim2real_mode': 'sim',
    'nlp_backend': 'tfidf',
    'max_ticks': 30,
    'log_latency': 2,
}


def _scripted_actions(env, rng):
    actions = {}
    for agent in env.agents:
        space = env.action_space(agent)
        actions[agent] = np.array(
            [rng.integers(0, n) for n in space.nvec], dtype=np.int64
        )
    return actions


class _ResolveSpy:
    """Monkey-patches ConflictResolutionEngine.resolve to capture outputs.

    resolve is a @staticmethod; we keep the descriptor on both patch and
    revert so we don't accidentally turn it into an instance method.
    """

    def __init__(self):
        self.last_resolved: dict = {}
        self._orig_func = ConflictResolutionEngine.resolve  # unwrapped

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
        parallel_env_module.ConflictResolutionEngine.resolve = staticmethod(
            self._orig_func
        )


def _flatten_deltas(resolved_effects) -> dict:
    """Flatten the per-agent state_deltas dicts into one dict for bulk apply.

    Last-writer-wins on key collision — matches the legacy semantics where
    parallel_env iterates resolved_effects.items() and calls apply_delta in
    that iteration order (Python 3.7+ dict preserves insertion order).
    """
    flat = {}
    for _agent, eff in resolved_effects.items():
        if eff is None or not eff.success:
            continue
        if isinstance(eff.state_deltas, dict):
            for k, v in eff.state_deltas.items():
                flat[k] = v
        # list-form (command objects) deferred — see functional.py module
        # docstring; the legacy engine still handles them via apply_delta.
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

            # Vectorized parity on every leaf the interpreter owns.
            np.testing.assert_array_equal(
                replayed.hosts.status,
                post.hosts.status,
                err_msg='status array diverged from legacy step',
            )
            np.testing.assert_array_equal(
                replayed.hosts.privilege,
                post.hosts.privilege,
                err_msg='privilege array diverged from legacy step',
            )
            np.testing.assert_array_equal(
                replayed.hosts.compromised_by_id,
                post.hosts.compromised_by_id,
                err_msg='compromised_by_id array diverged from legacy step',
            )
            np.testing.assert_array_equal(
                replayed.hosts.edr_active, post.hosts.edr_active
            )
            checked_steps += 1

    assert checked_steps > 0, 'episode terminated immediately; nothing checked'
