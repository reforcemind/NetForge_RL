import pytest

from benchmarks.self_play import _expected, population_tournament


@pytest.mark.fast
def test_expected_is_symmetric_at_equal_rating():
    assert _expected(1000, 1000) == pytest.approx(0.5)
    assert _expected(1200, 1000) > 0.5


@pytest.mark.integration
def test_tournament_produces_full_ladder():
    res = population_tournament(scenarios=['ransomware'], seeds=[0, 1], max_ticks=80)
    names = {e['policy'] for e in res['ladder']}
    assert {'random-red', 'heuristic-red', 'killchain-red'} <= names
    assert {'random-blue', 'heuristic-blue'} <= names
    # Ratings are a conserved-ish zero-sum ladder around the base.
    assert len(res['matches']) == 3 * 2
    for e in res['ladder']:
        assert 800 <= e['rating'] <= 1200
