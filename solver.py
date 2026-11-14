"""Per-task pipeline. Placeholder until the pipeline exists: every task gets the fallback answer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from arena.client import ArenaClient, Outcome, Task
from arena.llm import CostTracker


@dataclass(frozen=True, slots=True)
class Answer:
    outcome: Outcome
    answer: Any
    sources: list[dict] = field(default_factory=list)


FALLBACK = Answer("UNSUPPORTED", None)


def solve(client: ArenaClient, run_id: str, task: Task, tracker: CostTracker) -> Answer:
    return FALLBACK
