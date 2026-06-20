import pytest
from tests.parity.trajectory_fingerprint import roll_trajectory

GOLDEN_FINGERPRINT = 'abd164a5ef403918dced139e96010322625ac3a31e3af46fd7d5ad7b669912a2'
GOLDEN_SEED = 42
GOLDEN_MAX_TICKS = 25
GOLDEN_SCENARIO = 'ransomware'


@pytest.mark.integration
def test_legacy_backend_matches_golden_fingerprint() -> None:
    traj = roll_trajectory(
        seed=GOLDEN_SEED, max_ticks=GOLDEN_MAX_TICKS, scenario=GOLDEN_SCENARIO
    )
    actual = traj.fingerprint()
    assert actual == GOLDEN_FINGERPRINT, (
        f'Legacy backend trajectory fingerprint drifted.\n  expected: {GOLDEN_FINGERPRINT}\n  actual:   {actual}\nIf this drift is intentional, update GOLDEN_FINGERPRINT and add a changelog entry. Otherwise, bisect the offending commit.'
    )


@pytest.mark.fast
def test_fingerprint_is_deterministic_within_a_run() -> None:
    a = roll_trajectory(seed=7, max_ticks=10).fingerprint()
    b = roll_trajectory(seed=7, max_ticks=10).fingerprint()
    assert a == b, 'Trajectory fingerprint must be stable across re-rolls.'
