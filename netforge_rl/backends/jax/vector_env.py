from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from netforge_rl.backends.jax.action_codes import (  # noqa: F401 — re-exported
    BLUE_ANALYZE,
    BLUE_CONFIGURE_ACL,
    BLUE_DEPLOY_DECOY,
    BLUE_DEPLOY_HONEYTOKEN,
    BLUE_ISOLATE,
    BLUE_MISINFORM_APACHE,
    BLUE_MISINFORM_SSHD,
    BLUE_MISINFORM_TOMCAT,
    BLUE_MONITOR,
    BLUE_REMOVE,
    BLUE_RESTORE,
    BLUE_RESTORE_FROM_BACKUP,
    BLUE_ROTATE_KERBEROS,
    BLUE_SAT,
    EXFIL_PER_HOST,
    N_BLUE_ACTIONS,
    N_RED_ACTIONS,
    RED_COMPROMISE,
    RED_DISCOVER_REMOTE_SYSTEMS,
    RED_DUMP_LSASS,
    RED_EXFILTRATE,
    RED_EXPLOIT_BLUEKEEP,
    RED_EXPLOIT_ETERNALBLUE,
    RED_EXPLOIT_HTTP_RFI,
    RED_IMPACT,
    RED_IP_FRAGMENTATION,
    RED_JUICY_POTATO,
    RED_KILL_PROCESS,
    RED_KINETIC,
    RED_NETWORK_SCAN,
    RED_PASS_THE_HASH,
    RED_PASS_THE_TICKET,
    RED_PRIVESC,
    RED_RECON,
    RED_SHARE_INTEL,
    RED_SPEARPHISHING,
    RED_V4L2,
    SAT_DROP,
)
from netforge_rl.backends.jax.scenario_config import (  # noqa: F401 — re-exported
    SCENARIO_APT,
    SCENARIO_CLOUD,
    SCENARIO_IDS,
    SCENARIO_IOT,
    SCENARIO_OT,
    SCENARIO_RANSOMWARE,
)
from netforge_rl.backends.jax.state import JaxEnvState
from netforge_rl.backends.jax.transition import (  # noqa: F401 — re-exported
    scenario_done,
    single_env_step,
)


class BatchedActions(NamedTuple):
    red_target_idx: jax.Array
    blue_target_idx: jax.Array
    red_attempt: jax.Array
    blue_attempt: jax.Array
    red_action_type: jax.Array | None = None
    blue_action_type: jax.Array | None = None


@dataclass(frozen=True)
class VectorEnvSpec:
    n_hosts: int
    n_red: int
    n_blue: int
    scenario: int = SCENARIO_RANSOMWARE
    horizon: int = 200


def _default_action_types(
    actions: BatchedActions, spec: VectorEnvSpec
) -> BatchedActions:
    if actions.red_action_type is not None and actions.blue_action_type is not None:
        return actions
    batch = actions.red_target_idx.shape[0]
    return actions._replace(
        red_action_type=actions.red_action_type
        if actions.red_action_type is not None
        else jnp.zeros((batch, spec.n_red), dtype=jnp.int8),
        blue_action_type=actions.blue_action_type
        if actions.blue_action_type is not None
        else jnp.zeros((batch, spec.n_blue), dtype=jnp.int8),
    )


def make_vector_step(spec: VectorEnvSpec):
    """Return the compiled, batched transition step for this spec."""
    per_env = partial(single_env_step, spec=spec)
    batched = jax.vmap(per_env)

    @jax.jit
    def _step_impl(state, rt, bt, ra, ba, rat, bat):
        return batched(state, rt, bt, ra, ba, rat, bat)

    def step_fn(state, actions):
        actions = _default_action_types(actions, spec)
        return _step_impl(
            state,
            actions.red_target_idx,
            actions.blue_target_idx,
            actions.red_attempt,
            actions.blue_attempt,
            actions.red_action_type,
            actions.blue_action_type,
        )

    return step_fn


def random_actions(
    spec: VectorEnvSpec, batch_size: int, key: jax.Array
) -> BatchedActions:
    k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
    return BatchedActions(
        red_target_idx=jax.random.randint(
            k1, (batch_size, spec.n_red), 0, spec.n_hosts, dtype=jnp.int32
        ),
        blue_target_idx=jax.random.randint(
            k2, (batch_size, spec.n_blue), 0, spec.n_hosts, dtype=jnp.int32
        ),
        red_attempt=jax.random.bernoulli(k3, p=0.5, shape=(batch_size, spec.n_red)),
        blue_attempt=jax.random.bernoulli(k4, p=0.5, shape=(batch_size, spec.n_blue)),
        red_action_type=jax.random.randint(
            k5, (batch_size, spec.n_red), 0, N_RED_ACTIONS, dtype=jnp.int8
        ),
        blue_action_type=jax.random.randint(
            k6, (batch_size, spec.n_blue), 0, N_BLUE_ACTIONS, dtype=jnp.int8
        ),
    )


def initial_batched_state(template: JaxEnvState, batch_size: int) -> JaxEnvState:
    """Tile state across the batch dimension."""

    def tile(x):
        if isinstance(x, (jax.Array, np.ndarray)):
            return jnp.broadcast_to(jnp.asarray(x), (batch_size,) + tuple(x.shape))
        return jnp.broadcast_to(jnp.asarray(x), (batch_size,))

    return jax.tree_util.tree_map(tile, template)


def jax_siem_features(hosts) -> jax.Array:
    """Per-host SIEM alert signal in [0, 1], computed in JAX from state — the
    vectorized-backend analogue of the Python SIEM pipeline (jit/vmap-safe)."""
    compromised = (hosts.compromised_by_id >= 0).astype(jnp.float32)
    privileged = (hosts.privilege > 0).astype(jnp.float32)
    honeytoken = hosts.contains_honeytokens.astype(jnp.float32) * compromised
    decoy = (hosts.decoy > 0).astype(jnp.float32)
    edr = hosts.edr_active.astype(jnp.float32)

    alert = 0.4 * compromised + 0.3 * privileged + 0.5 * honeytoken + 0.2 * decoy
    # EDR sharpens detection confidence on the hosts it covers.
    alert = alert * (1.0 + 0.25 * edr)
    return jnp.clip(alert, 0.0, 1.0)
