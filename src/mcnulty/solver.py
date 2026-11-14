"""Per-task pipeline. Placeholder until the pipeline exists: every task gets the fallback submission."""

from mcnulty.arena import ArenaClient, Submission, Task
from mcnulty.llm import OpenRouterClient
from mcnulty.recorder import Recorder

FALLBACK = Submission(outcome="UNSUPPORTED", answer=None)


def solve(
    rec: Recorder, arena: ArenaClient, llm: OpenRouterClient, run_id: str, task: Task
):
    return FALLBACK
