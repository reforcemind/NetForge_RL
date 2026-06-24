from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class StepRecord:
    tick: int
    agent_id: str
    action_name: str
    target_ip: Optional[str]
    success: bool
    reward: float


@dataclass
class TrajectoryRecorder:
    """Records per-step (tick, agent, action, target, success, reward) for kill-chain analysis."""

    scenario: str = ''
    seed: int = 0
    _records: list[StepRecord] = field(default_factory=list, repr=False)

    def reset(self, scenario: str = '', seed: int = 0) -> None:
        self.scenario = scenario
        self.seed = seed
        self._records.clear()

    def record_step(
        self,
        tick: int,
        agent_id: str,
        action_name: str,
        target_ip: Optional[str],
        success: bool,
        reward: float,
    ) -> None:
        self._records.append(
            StepRecord(
                tick=tick,
                agent_id=agent_id,
                action_name=action_name,
                target_ip=target_ip,
                success=success,
                reward=reward,
            )
        )

    def kill_chain(self, agent_id: str) -> list[StepRecord]:
        """Ordered successful actions for one agent — the kill chain."""
        return [r for r in self._records if r.agent_id == agent_id and r.success]

    def summary(self) -> dict:
        if not self._records:
            return {}
        agents = sorted({r.agent_id for r in self._records})
        result = {
            'scenario': self.scenario,
            'seed': self.seed,
            'total_steps': len(self._records),
            'ticks': self._records[-1].tick if self._records else 0,
            'agents': {},
        }
        for agent in agents:
            agent_records = [r for r in self._records if r.agent_id == agent]
            successes = [r for r in agent_records if r.success]
            result['agents'][agent] = {
                'actions_taken': len(agent_records),
                'successes': len(successes),
                'total_reward': round(sum(r.reward for r in agent_records), 4),
                'kill_chain': [
                    {'tick': r.tick, 'action': r.action_name, 'target': r.target_ip}
                    for r in successes
                ],
            }
        return result

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'scenario': self.scenario,
            'seed': self.seed,
            'records': [asdict(r) for r in self._records],
        }
        path.write_text(json.dumps(payload, indent=2))
        return path

    @classmethod
    def load(cls, path: str | Path) -> TrajectoryRecorder:
        data = json.loads(Path(path).read_text())
        rec = cls(scenario=data.get('scenario', ''), seed=data.get('seed', 0))
        for r in data.get('records', []):
            rec._records.append(StepRecord(**r))
        return rec

    def __len__(self) -> int:
        return len(self._records)
