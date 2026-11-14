"""``mcnulty run`` and ``mcnulty collect``: a live run on the Task API. Each task is solved, its checked actions are
performed, and it is submitted at once; one stdout line per task (only stdout comes back from the final container).
"""

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pydantic import Field

from mcnulty.agent.actions import tool_args
from mcnulty.agent.loop import solve
from mcnulty.agent.submit import FALLBACK
from mcnulty.clients.arena import MAX_NOTE, ArenaClient, ArenaError, RunKind, Task, git_meta
from mcnulty.clients.openrouter import BudgetExhausted, CostTracker, OpenRouterClient
from mcnulty.commands.task_filter import TaskFilter
from mcnulty.frozen import Frozen
from mcnulty.record.events import CliStarted, RunFinished, TaskCrashed, TaskFinished
from mcnulty.record.log import EventLog, Recorder
from mcnulty.settings import Settings


class RunArgs(Frozen):
    kind: RunKind
    note: str | None = Field(default=None, max_length=MAX_NOTE)
    seed: int | None = None
    only: TaskFilter = TaskFilter()
    parallel: int = Field(default=12, ge=1)
    collect: bool = False


def run_tasks(settings: Settings, args: RunArgs):
    run_dir = Path("runs") / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    rec = Recorder(events=EventLog(run_dir=run_dir, secrets=settings.secrets))
    rec.log(CliStarted(argv=tuple(sys.argv[1:])))
    arena = ArenaClient.connect(settings)
    llm = OpenRouterClient.connect(settings, CostTracker())
    try:
        tools = arena.tools(rec)
        meta = git_meta().model_copy(update={"note": args.note})
        run = arena.start_run(rec, args.kind, args.seed if args.kind == "dev" else None, meta)
    except ArenaError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    (run_dir / "run.json").write_text(run.model_dump_json(indent=1), encoding="utf-8")
    (run_dir / "tools.json").write_text("[" + ",".join(t.model_dump_json() for t in tools) + "]", encoding="utf-8")

    # After a container restart the final continues the same run: skip what is already submitted.
    done = {t.task_id for t in arena.run_status(rec, run.run_id).tasks if t.submitted}
    tasks = [t for t in args.only.select(run.tasks) if t.task_id not in done]
    note = f", note: {args.note}" if args.note else ""
    print(
        f"run {run.run_id} ({run.kind}, theme={run.theme}, seed={run.seed}), {len(tasks)} tasks, "
        f"{len(done)} already submitted, dir={run_dir}{note}"
    )
    manifest = arena.run_manifest(rec, run.run_id)
    (run_dir / "manifest.json").write_text(manifest.model_dump_json(indent=1), encoding="utf-8")
    manifest_files = {t.task_id: t.files for t in manifest.tasks}
    stats = arena.prefetch(rec, manifest, run.run_id)
    print(f"prefetch: {stats.blobs} files, {stats.downloaded} downloaded ({stats.bytes / 1e6:.1f} MB)")

    def work(task: Task):
        task_rec = rec.for_task(task.task_id)
        t0 = time.time()
        try:
            solution = solve(task_rec, llm, task, manifest_files[task.task_id], arena.cache)
        except BudgetExhausted:
            solution = FALLBACK
        except Exception as e:  # one task must never kill the run; always submit something
            task_rec.log(TaskCrashed.from_exception(e))
            solution = FALLBACK
        submission = solution.submission
        for vetted in solution.actions:
            if vetted.blocked:
                continue
            try:
                arena.call_tool(task_rec, run.run_id, task.task_id, vetted.action.tool, tool_args(vetted.action))
            except ArenaError as e:
                task_rec.log(TaskCrashed.from_exception(e))
        try:
            result = arena.submit(task_rec, run.run_id, task.task_id, submission)
            score, feedback = result.score, result.feedback
        except ArenaError as e:
            task_rec.log(TaskCrashed.from_exception(e))
            score, feedback = None, (f"submit failed: {e}",)
        finished = TaskFinished(
            outcome=submission.outcome,
            seconds=round(time.time() - t0, 1),
            run_cost_usd=llm.tracker.total,
            score=score,
            feedback=feedback,
        )
        task_rec.log(finished)
        # one short line per task: only stdout comes back from the final container
        head = f"{task.task_id} L{task.level} {submission.outcome} {finished.seconds:.0f}s"
        print(f"{head} ${finished.run_cost_usd:.4f} {' | '.join(feedback)}")

    if not args.collect:
        with ThreadPoolExecutor(max_workers=args.parallel) as ex:
            list(ex.map(work, tasks))

    finish = arena.finish(rec, run.run_id)
    (run_dir / "finish.json").write_text(finish.model_dump_json(indent=1), encoding="utf-8")
    summary = {"status", "tasks_submitted", "tasks_total", "accuracy", "cost_usd", "cost_score", "final_score"}
    print(f"finish: {finish.model_dump_json(include=summary)}")
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
    print(f"local cost ${t.total:.4f} in {t.calls} calls ({t.prompt_tokens} in / {t.completion_tokens} out)")
    return 0
