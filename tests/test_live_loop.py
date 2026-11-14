"""The agent loop end to end with real model calls and the real sandbox, on one task of a saved dev run.

Runs only inside the Docker image with the keys in the environment and runs/ and .cache/ mounted (AGENTS.md,
Commands); skipped elsewhere. The model is not deterministic, so this checks the plumbing, not the answer.
"""

import os
from pathlib import Path

import pytest

from mcnulty.agent.loop import MAX_STEPS, solve
from mcnulty.agent.sandbox import RUNNER, SandboxPool
from mcnulty.blobs import BlobCache
from mcnulty.clients.openrouter import CostTracker, OpenRouterClient
from mcnulty.commands.evaluate import SavedRun
from mcnulty.record.log import EventLog, Recorder
from mcnulty.settings import REPO_ROOT, load_settings

SAVED = REPO_ROOT / "runs" / "20261114T101003Z"
pytestmark = pytest.mark.skipif(
    not (RUNNER.exists() and "OPENROUTER_API_KEY" in os.environ and (SAVED / "finish.json").exists()),
    reason="live: needs the Docker image, the keys and the saved run",
)


@pytest.mark.parametrize("task_id", ["t01", "t07"])
def test_a_real_task_ends_in_a_checked_submission(tmp_path: Path, task_id: str):
    settings = load_settings()
    saved = SavedRun.load(SAVED)
    task = next(t for t in saved.run.tasks if t.task_id == task_id)
    llm = OpenRouterClient.connect(settings, CostTracker(cap=0.10))
    rec = Recorder(events=EventLog(run_dir=tmp_path / "run", secrets=settings.secrets))
    try:
        solution = solve(rec, llm, task, saved.files(task_id), BlobCache(root=settings.blob_cache), SandboxPool())
    finally:
        llm.http.close()
    events = (tmp_path / "run" / "events.jsonl").read_text(encoding="utf-8")
    assert '"kind":"sandbox"' in events  # the model ran code
    assert solution.steps < MAX_STEPS  # it finished by submit(...), not by the step limit
    assert solution.submission.outcome in {"OK", "DENIED_SECURITY", "NEEDS_CLARIFICATION", "UNSUPPORTED"}
