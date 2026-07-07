import pytest

from netforge_rl.baselines.policies import HeuristicBluePolicy, RandomPolicy
from netforge_rl.diagnostics.capability_card import capability_card


@pytest.mark.integration
def test_card_has_all_capabilities():
    card = capability_card(lambda: HeuristicBluePolicy(seed=0), seeds=(0,))
    assert set(card['capabilities']) == {
        'memory',
        'attention',
        'temporal',
        'precision',
        'safety',
        'generalization',
    }
    assert 0.0 <= card['overall'] <= 1.0


@pytest.mark.integration
def test_card_ranks_heuristic_above_random():
    hb = capability_card(lambda: HeuristicBluePolicy(seed=0), seeds=(0, 1))
    rnd = capability_card(lambda: RandomPolicy(seed=0), seeds=(0, 1))
    assert hb['overall'] > rnd['overall']


@pytest.mark.integration
def test_card_writes_artifacts(tmp_path):
    capability_card(
        lambda: HeuristicBluePolicy(seed=0),
        seeds=(0,),
        out_dir=str(tmp_path),
        name='hb',
    )
    assert (tmp_path / 'hb_card.json').exists()
    # PNG only if matplotlib is installed; JSON is always written.
