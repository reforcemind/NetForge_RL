from dataclasses import dataclass

import numpy as np

from netforge_rl.semantic.la_wrapper import state_to_text
from netforge_rl.semantic.parser import parse_action


@dataclass
class RolloutBatch:
    queries: list
    responses: list
    rewards: list
    invalid: int


class LMPolicyAdapter:
    """Bridge a NetForge env onto trl's (query, response, reward) protocol.

    queries() emits one SIEM prompt; step(responses) parses the reply,
    advances the env, returns a RolloutBatch. Invalid replies incur
    invalid_penalty so the policy gradient gets signal.
    """

    def __init__(
        self,
        env,
        controlled_agent,
        *,
        seed=0,
        invalid_penalty=-1.0,
        max_hosts_in_prompt=8,
    ):
        self.env = env
        self.agent = controlled_agent
        self.invalid_penalty = invalid_penalty
        self.max_hosts_in_prompt = max_hosts_in_prompt
        self._last_query = None
        self.env.reset(seed=seed)

    def queries(self):
        snap = self.env.to_envstate()
        text = state_to_text(snap, self.agent, max_hosts=self.max_hosts_in_prompt)
        self._last_query = text
        return [text]

    def step(self, responses):
        assert self._last_query is not None, 'call queries() before step()'
        target_ips = sorted(self.env.global_state.all_hosts.keys())
        response = responses[0]
        parsed = parse_action(response, self.agent, target_ips)

        actions = {a: np.array([0, 0], dtype=np.int64) for a in self.env.agents}
        invalid = 0
        if parsed is None:
            invalid = 1
        else:
            actions[self.agent] = np.array(parsed, dtype=np.int64)

        _, rewards, term, trunc, _ = self.env.step(actions)
        reward = float(rewards.get(self.agent, 0.0))
        if invalid:
            reward += self.invalid_penalty

        batch = RolloutBatch(
            queries=[self._last_query],
            responses=[response],
            rewards=[reward],
            invalid=invalid,
        )

        if (all(term.values()) or all(trunc.values())) and self.env.agents == []:
            self.env.reset()

        return batch
