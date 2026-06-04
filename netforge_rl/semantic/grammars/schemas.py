from netforge_rl.semantic.action_menu import action_menu


def anthropic_tool_schema(agent_id, target_ips):
    """Anthropic tool_use schema constraining replies to a legal (action, ip) pair."""
    menu = action_menu(agent_id)
    return {
        'name': 'submit_action',
        'description': (
            'Choose one MITRE-style cyber action against one host on the network. '
            'Only the enumerated action_id values and target_ip values are legal.'
        ),
        'input_schema': {
            'type': 'object',
            'properties': {
                'action_id': {
                    'type': 'integer',
                    'enum': list(menu.keys()),
                    'description': 'Numeric ID of the chosen action (see legend).',
                },
                'target_ip': {
                    'type': 'string',
                    'enum': list(target_ips),
                    'description': 'IP address of the target host.',
                },
            },
            'required': ['action_id', 'target_ip'],
        },
    }


def openai_tool_schema(agent_id, target_ips):
    """OpenAI tool_choice / function-call JSON schema for the same contract."""
    menu = action_menu(agent_id)
    return {
        'type': 'function',
        'function': {
            'name': 'submit_action',
            'description': 'Submit one cyber action against one host.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'action_id': {'type': 'integer', 'enum': list(menu.keys())},
                    'target_ip': {'type': 'string', 'enum': list(target_ips)},
                },
                'required': ['action_id', 'target_ip'],
                'additionalProperties': False,
            },
        },
    }


def vllm_grammar(agent_id, target_ips):
    """Lark grammar string for xgrammar / outlines / lm-format-enforcer.

    Forces the model to emit exactly `ACTION <id> TARGET <ip>` where <id> and
    <ip> come from the legal sets.
    """
    ids = ' | '.join(f'"{i}"' for i in action_menu(agent_id))
    ips = ' | '.join(f'"{ip}"' for ip in target_ips)
    return (
        f'start: "ACTION " action_id " TARGET " target_ip\n'
        f'action_id: {ids}\n'
        f'target_ip: {ips}\n'
    )
