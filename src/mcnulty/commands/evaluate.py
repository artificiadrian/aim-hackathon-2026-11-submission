"""``mcnulty eval``: run the solver offline on a saved dev run and grade each task (``grading``) against its revealed
answers. Model spend stops at a hard cap. Runs only inside the Docker image (the sandbox).
"""

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pydantic import Field

from mcnulty.agent.loop import Transcripts, solve
from mcnulty.agent.sandbox import SandboxPool
from mcnulty.agent.submit import FALLBACK
from mcnulty.blobs import BlobCache
from mcnulty.clients.arena import MAX_NOTE, Run, RunManifest, RunStatus, git_meta
from mcnulty.clients.openrouter import BudgetExhausted, CostTracker, OpenRouterClient
from mcnulty.commands.grading import grade
from mcnulty.commands.task_filter import TaskFilter
from mcnulty.frozen import FromText, Frozen
from mcnulty.record.events import CliStarted, RunFinished, TaskCrashed, TaskScored
from mcnulty.record.log import EventLog, Recorder
from mcnulty.settings import Settings


class EvalArgs(FromText):
    run_dir: Path
    note: str | None = Field(default=None, max_length=MAX_NOTE)
    only: TaskFilter = TaskFilter()
    max_usd: float = Field(gt=0, le=2.0)


class SavedRun(Frozen):
    """A finished dev run in ``runs/``: its tasks, files and revealed answers."""

    run: Run
    manifest: RunManifest
    finish: RunStatus

    @classmethod
    def load(cls, run_dir: Path):
        def read(name: str):
            return (run_dir / name).read_text(encoding="utf-8")

        return cls(
            run=Run.model_validate_json(read("run.json")),
            manifest=RunManifest.model_validate_json(read("manifest.json")),
            finish=RunStatus.model_validate_json(read("finish.json")),
        )

    def files(self, task_id: str):
        return next(t.files for t in self.manifest.tasks if t.task_id == task_id)

    def expected(self, task_id: str):
        return next(t for t in self.finish.tasks if t.task_id == task_id)


class EvalSummary(Frozen):
    run_dir: str
    seed: int | None
    tasks: int
    accuracy: float  # mean task score over the evaluated tasks
    cost_usd: float
    scores: dict[str, TaskScored]


def run_eval(rec: Recorder, llm: OpenRouterClient, blobs: BlobCache, saved: SavedRun, task_ids: tuple[str, ...]):
    """Solve and grade ``task_ids`` of ``saved``; one stdout line per task. Model spend stops at the tracker's cap."""

    pool = SandboxPool()
    transcripts = Transcripts()  # images shared by the tasks are read once per run

    def work(task_id: str):
        task = next(t for t in saved.run.tasks if t.task_id == task_id)
        task_rec = rec.for_task(task_id)
        task_llm = llm.for_task(len(task_ids))
        t0 = time.time()
        try:
            solution = solve(task_rec, task_llm, task, saved.files(task_id), blobs, pool, transcripts=transcripts)
        except BudgetExhausted:
            solution = FALLBACK
        except Exception as e:  # one task must never stop the evaluation
            task_rec.log(TaskCrashed.from_exception(e))
            solution = FALLBACK
        spent = task_llm.tracker
        scored = grade(solution, saved.expected(task_id)).model_copy(
            update={
                "steps": solution.steps,
                "calls": spent.calls,
                "prompt_tokens": spent.prompt_tokens,
                "cost_usd": round(spent.total, 6),
            }
        )
        task_rec.log(scored)
        line = (
            f"{task_id} L{task.level} {scored.outcome}/{scored.expected_outcome}",
            f"answer={scored.answer_score:.2f} sources={scored.sources_hit}/{scored.sources_expected}",
            f"cited={scored.sources_cited} score={scored.score:.2f} est={scored.score_est:.2f}",
            f"{time.time() - t0:.0f}s steps={solution.steps} calls={spent.calls}",
            f"${spent.total:.4f} (run ${llm.tracker.total:.4f})",
        )
        print(" ".join(line))
        return task_id, scored

    with ThreadPoolExecutor(max_workers=4) as ex:
        scores = dict(ex.map(work, task_ids))
    t = llm.tracker
    rec.log(
        RunFinished(
            cost_usd=t.total,
            calls=t.calls,
            prompt_tokens=t.prompt_tokens,
            completion_tokens=t.completion_tokens,
            by_model=t.by_model,
        )
    )
    return EvalSummary(
        run_dir=str(rec.events.run_dir),
        seed=saved.run.seed,
        tasks=len(scores),
        accuracy=round(sum(s.score for s in scores.values()) / len(scores), 4) if scores else 0.0,
        cost_usd=round(t.total, 6),
        scores=scores,
    )


def eval_tasks(settings: Settings, args: EvalArgs):
    if not (args.run_dir / "finish.json").exists():
        print(f"error: {args.run_dir} has no finish.json (not a finished dev run)", file=sys.stderr)
        return 1
    saved = SavedRun.load(args.run_dir)
    run_dir = Path("runs") / (time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-eval")
    rec = Recorder(events=EventLog(run_dir=run_dir, secrets=settings.secrets))
    rec.log(CliStarted(argv=tuple(sys.argv[1:])))
    blobs = BlobCache(root=settings.blob_cache)
    task_ids = tuple(t.task_id for t in args.only.select(saved.run.tasks))
    missing = {sha for i in task_ids for sha in saved.files(i).values() if not (blobs.root / sha).exists()}
    if missing:
        print(f"error: {len(missing)} files of these tasks are not in {blobs.root} (collect run?)", file=sys.stderr)
        return 1
    meta = git_meta().model_copy(update={"note": args.note})
    (run_dir / "eval_args.json").write_text(
        json.dumps({"args": args.model_dump(mode="json"), "meta": meta.model_dump()}, indent=1), encoding="utf-8"
    )
    print(f"eval of {args.run_dir} (seed={saved.run.seed}), {len(task_ids)} tasks, cap ${args.max_usd}, dir={run_dir}")
    llm = OpenRouterClient.connect(settings, CostTracker(cap=args.max_usd))
    summary = run_eval(rec, llm, blobs, saved, task_ids)
    (run_dir / "eval.json").write_text(summary.model_dump_json(indent=1), encoding="utf-8")
    print(f"accuracy {summary.accuracy:.3f} over {summary.tasks} tasks, cost ${summary.cost_usd:.4f}")
    return 0
