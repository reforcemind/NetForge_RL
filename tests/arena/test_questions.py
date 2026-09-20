import json

import pytest

from netforge_rl.arena.questions import (
    QUESTIONS,
    get_question,
    list_questions,
    run_question,
)
from netforge_rl.arena.submissions import evaluate_submission, load_submission
from netforge_rl.cli import main


@pytest.mark.fast
def test_catalog_has_runnable_and_protocol_questions():
    ids = {q.id for q in QUESTIONS}
    assert 'belief-vs-oracle' in ids
    assert 'gnn-belief' in ids
    assert get_question('reward-design').status == 'runnable'
    assert get_question('league-psro').status == 'protocol'
    rows = list_questions(runnable_only=True)
    assert all(r['status'] == 'runnable' for r in rows)
    assert len(rows) >= 8


@pytest.mark.fast
def test_belief_vs_oracle_reports_information_gap():
    payload = run_question('belief-vs-oracle', seeds=(0,), max_ticks=4)
    answer = payload['answer']
    assert 0.0 <= answer['visible_ratio'] <= 1.0
    assert answer['compromise_recall'] == 0.0
    assert answer['oracle_compromised_nodes'] >= 1


@pytest.mark.fast
def test_protocol_question_does_not_pretend_to_run():
    payload = run_question('gnn-belief')
    assert payload['status'] == 'protocol'
    assert 'answer' not in payload


@pytest.mark.fast
def test_cli_questions_list(capsys):
    assert main(['questions']) == 0
    out = capsys.readouterr().out
    assert 'belief-vs-oracle' in out


@pytest.mark.fast
def test_cli_questions_json(capsys):
    assert main(['questions', '--json']) == 0
    payload = json.loads(capsys.readouterr().out)
    assert any(q['id'] == 'time-mode' for q in payload)


@pytest.mark.integration
def test_reward_design_question_runs():
    payload = run_question('reward-design', seeds=(0,), max_ticks=6)
    variants = payload['answer']['variants']
    assert set(variants) == {'default', 'sla_only', 'fp_penalized'}


@pytest.mark.integration
def test_example_submission_evaluates_without_reward_ranking():
    sub = load_submission('examples/submissions/heuristic_blue.py')
    result = evaluate_submission(
        sub,
        scenarios=('ransomware',),
        seeds=(0,),
        max_ticks=6,
        official=False,
    )
    assert result['name'] == 'heuristic-blue'
    assert 'mission_success' in result['headline']
    assert 'mean_blue_return' in result['headline']
    assert any('not mean reward' in w for w in result['warnings'])
