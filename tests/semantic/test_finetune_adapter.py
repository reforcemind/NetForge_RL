import pytest
from netforge_rl.semantic.finetune import LMPolicyAdapter


@pytest.mark.integration
def test_adapter_round_trip(env_sim) -> None:
    adapter = LMPolicyAdapter(env_sim, controlled_agent='blue_dmz', seed=42)
    q = adapter.queries()
    assert len(q) == 1 and 'NetForge SIEM' in q[0]
    target = next(
        (ip for ip in env_sim.global_state.all_hosts if not ip.startswith('169.254.'))
    )
    batch = adapter.step([f'ACTION 0 TARGET {target}'])
    assert batch.queries == q
    assert batch.invalid == 0
    assert len(batch.rewards) == 1


@pytest.mark.integration
def test_adapter_penalises_invalid_responses(env_sim) -> None:
    adapter = LMPolicyAdapter(
        env_sim, controlled_agent='blue_dmz', seed=42, invalid_penalty=-7.5
    )
    adapter.queries()
    batch_bad = adapter.step(['nope'])
    assert batch_bad.invalid == 1
    adapter2 = LMPolicyAdapter(
        env_sim, controlled_agent='blue_dmz', seed=42, invalid_penalty=0.0
    )
    adapter2.queries()
    batch_baseline = adapter2.step(['nope'])
    assert batch_bad.rewards[0] == pytest.approx(batch_baseline.rewards[0] - 7.5)
