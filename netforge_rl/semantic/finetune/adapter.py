"""Bridge between NetForge's PettingZoo env and HuggingFace trl.

A ``RolloutBatch`` is exactly what ``trl.PPOTrainer.step`` consumes:
parallel lists of ``queries``, ``responses``, and ``rewards``. The
adapter does not depend on ``trl`` or ``transformers`` — those are the
caller's concern — so it stays cheap to test and import.

Usage from a trainer loop:

    adapter = LMPolicyAdapter(env, controlled_agent='blue_dmz')
    for epoch in range(N):
        queries = adapter.queries()
        responses = trainer.generate(queries)
        batch = adapter.step(responses)
        trainer.step(batch.queries, batch.responses, batch.rewards)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from netforge_rl.semantic.la_wrapper import state_to_text
from netforge_rl.semantic.parser import parse_action


@dataclass
class RolloutBatch:
    queries: list[str]
    responses: list[str]
    rewards: list[float]
    invalid: int


class LMPolicyAdapter:
    """Bridge a NetForge env onto trl's (query, response, reward) protocol.

    ``controlled_agent`` is the agent whose policy is the LLM being trained;
    other agents take no-op actions. Reward is the per-step env reward for
    the controlled agent. Invalid responses are penalised by
    ``invalid_penalty`` to give the policy gradient a learning signal.
    """

    def __init__(
        self,
        env,
        controlled_agent: str,
        *,
        seed: int = 0,
        invalid_penalty: float = -1.0,
        max_hosts_in_prompt: int = 8,
    ):
        self.env = env
        self.agent = controlled_agent
        self.invalid_penalty = invalid_penalty
        self.max_hosts_in_prompt = max_hosts_in_prompt
        self._last_query: str | None = None
        self.env.reset(seed=seed)

    def queries(self) -> list[str]:
        """Single-element list of the current SIEM prompt for the agent."""
        snap = self.env.to_envstate()
        text = state_to_text(snap, self.agent, max_hosts=self.max_hosts_in_prompt)
        self._last_query = text
        return [text]

    def step(self, responses: list[str]) -> RolloutBatch:
        """Apply the LLM's response, step env once, return a RolloutBatch."""
        assert self._last_query is not None, 'call queries() before step()'
        target_ips = sorted(self.env.global_state.all_hosts.keys())
        response = responses[0]
        parsed = parse_action(response, self.agent, target_ips)

        actions: dict[str, np.ndarray] = {
            a: np.array([0, 0], dtype=np.int64) for a in self.env.agents
        }
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
