"""Per-task pipeline. Placeholder until the pipeline exists: every task gets the fallback submission.

The solver sees only the task and its files (path -> sha256 in the blob cache), never the Task API, so the same code
runs live (``mcnulty run``) and offline on saved runs (``mcnulty eval``).
"""

from mcnulty.arena import BlobCache, Submission, Task
from mcnulty.llm import OpenRouterClient
from mcnulty.recorder import Recorder

FALLBACK = Submission(outcome="UNSUPPORTED", answer=None)


def solve(rec: Recorder, llm: OpenRouterClient, task: Task, files: dict[str, str], blobs: BlobCache):
    return FALLBACK
