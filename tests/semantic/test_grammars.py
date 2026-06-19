import pytest

pytest.importorskip('PIL')
import pytest
from netforge_rl.semantic.grammars import (
    anthropic_tool_schema,
    openai_tool_schema,
    vllm_grammar,
)


@pytest.mark.fast
def test_anthropic_schema_lists_legal_actions():
    s = anthropic_tool_schema('blue_dmz', ['10.0.0.1', '10.0.0.2'])
    assert s['name'] == 'submit_action'
    props = s['input_schema']['properties']
    assert 0 in props['action_id']['enum']
    assert '10.0.0.1' in props['target_ip']['enum']


@pytest.mark.fast
def test_openai_schema_shape():
    s = openai_tool_schema('red_operator', ['10.0.0.1'])
    assert s['type'] == 'function'
    assert s['function']['parameters']['additionalProperties'] is False
    enum = s['function']['parameters']['properties']['action_id']['enum']
    assert 0 in enum


@pytest.mark.fast
def test_vllm_grammar_lark_starts_with_action():
    g = vllm_grammar('blue_dmz', ['10.0.0.1'])
    assert 'start:' in g
    assert '"ACTION "' in g
    assert '"10.0.0.1"' in g
    assert '"0"' in g


@pytest.mark.fast
def test_schemas_reject_when_no_target_ips():
    s = anthropic_tool_schema('blue_dmz', [])
    assert s['input_schema']['properties']['target_ip']['enum'] == []
