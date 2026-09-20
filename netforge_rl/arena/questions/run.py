from __future__ import annotations

from collections.abc import Sequence

from netforge_rl.arena.questions.catalog import BY_ID, QUESTIONS
from netforge_rl.arena.questions.runners import RUNNERS
from netforge_rl.arena.questions.types import ResearchQuestion


def get_question(qid: str) -> ResearchQuestion:
    try:
        return BY_ID[qid.strip().lower()]
    except KeyError:
        raise KeyError(
            f'Unknown question {qid!r}. Choose from: {", ".join(BY_ID)}'
        ) from None


def list_questions(*, runnable_only: bool = False) -> list[dict]:
    qs = (
        QUESTIONS
        if not runnable_only
        else (q for q in QUESTIONS if q.status == 'runnable')
    )
    return [q.as_dict() for q in qs]


def run_question(
    qid: str,
    *,
    seeds: Sequence[int] = (0,),
    max_ticks: int = 20,
) -> dict:
    q = get_question(qid)
    if q.status != 'runnable':
        return {
            'id': q.id,
            'status': 'protocol',
            'question': q.question,
            'protocol': q.protocol,
            'note': q.note,
        }
    payload = RUNNERS[q.id](seeds=tuple(seeds), max_ticks=int(max_ticks))
    return {
        'id': q.id,
        'title': q.title,
        'question': q.question,
        'status': 'runnable',
        'answer': payload,
    }


def run_all_runnable(
    *,
    seeds: Sequence[int] = (0,),
    max_ticks: int = 12,
    ids: Sequence[str] | None = None,
) -> list[dict]:
    chosen = (
        list(ids)
        if ids is not None
        else [q.id for q in QUESTIONS if q.status == 'runnable']
    )
    return [run_question(qid, seeds=seeds, max_ticks=max_ticks) for qid in chosen]


def format_catalog() -> str:
    lines = ['id                      status     question']
    for q in QUESTIONS:
        lines.append(f'{q.id:<24}{q.status:<11}{q.question}')
    return '\n'.join(lines)
