import numpy as np
import pytest

pytest.importorskip("PIL")

from netforge_rl.baselines.policies import RandomPolicy
from netforge_rl.diagnostics import (
    DiagnosticResult,
    MemoryProbe,
    NoisySIEM,
    all_diagnostics,
    run_diagnostic,
)


class _PerfectMemoryBlue:
    name = 'perfect_memory'

    def __init__(self, planted_ip):
        self.planted_ip = planted_ip

    def act(self, env, agent_id):
        ips = sorted(env.global_state.all_hosts.keys())
        if agent_id == 'blue_dmz' and self.planted_ip in ips:
            return np.array([0, ips.index(self.planted_ip)], dtype=np.int64)
        return np.array([0, 0], dtype=np.int64)


@pytest.mark.fast
def test_all_diagnostics_have_required_attrs():
    for d in all_diagnostics():
        assert d.name
        assert d.capability
        assert d.max_ticks > 0


@pytest.mark.integration
@pytest.mark.parametrize('diag_cls', [MemoryProbe, NoisySIEM])
def test_diagnostic_returns_normalized_score(diag_cls):
    result = run_diagnostic(diag_cls(), RandomPolicy(seed=0), seed=0)
    assert isinstance(result, DiagnosticResult)
    assert 0.0 <= result.score <= 1.0
    assert result.diagnostic == diag_cls().name


@pytest.mark.integration
def test_memory_probe_discriminates_random_vs_perfect():
    diag = MemoryProbe()
    env = diag.build_env(seed=0)
    diag.setup(env)
    perfect = _PerfectMemoryBlue(planted_ip=diag._planted_ip)
    r_perfect = run_diagnostic(MemoryProbe(), perfect, seed=0)
    r_random = run_diagnostic(MemoryProbe(), RandomPolicy(seed=7), seed=0)
    assert r_perfect.score > r_random.score
    assert r_perfect.score >= 0.9


@pytest.mark.fast
def test_diagnostic_is_deterministic():
    a = run_diagnostic(MemoryProbe(), RandomPolicy(seed=42), seed=0)
    b = run_diagnostic(MemoryProbe(), RandomPolicy(seed=42), seed=0)
    assert a.score == b.score
    assert a.details['planted_ip'] == b.details['planted_ip']
