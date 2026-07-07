import pytest

from netforge_rl.baselines.policies import HeuristicBluePolicy, RandomPolicy
from netforge_rl.diagnostics import all_diagnostics, run_diagnostic
from netforge_rl.diagnostics.base import DiagnosticResult


@pytest.mark.fast
def test_suite_has_six_distinct_capabilities():
    diags = all_diagnostics()
    assert len(diags) == 6
    caps = {d.capability for d in diags}
    assert caps == {
        'memory',
        'attention',
        'temporal',
        'precision',
        'safety',
        'generalization',
    }


@pytest.mark.integration
@pytest.mark.parametrize('idx', range(6))
def test_each_probe_runs_and_scores_in_unit_range(idx):
    diag = all_diagnostics()[idx]
    result = run_diagnostic(diag, HeuristicBluePolicy(seed=0), seed=0)
    assert isinstance(result, DiagnosticResult)
    assert 0.0 <= result.score <= 1.0
    assert result.capability == diag.capability


@pytest.mark.integration
def test_probes_are_discriminative():
    """A competent blue policy should beat random on the containment-style probes."""
    containment = {'memory', 'attention', 'temporal', 'safety'}
    for diag in all_diagnostics():
        if diag.capability not in containment:
            continue
        heuristic = run_diagnostic(diag, HeuristicBluePolicy(seed=0), seed=0).score
        rand = run_diagnostic(type(diag)(), RandomPolicy(seed=0), seed=0).score
        assert heuristic >= rand, f'{diag.name}: heuristic {heuristic} < random {rand}'


@pytest.mark.integration
def test_probe_scores_are_seed_deterministic():
    diag = all_diagnostics()[0]
    a = run_diagnostic(diag, HeuristicBluePolicy(seed=0), seed=1).score
    b = run_diagnostic(type(diag)(), HeuristicBluePolicy(seed=0), seed=1).score
    assert a == b
