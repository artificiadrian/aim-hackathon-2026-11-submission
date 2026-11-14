"""``mcnulty run`` and ``mcnulty collect``: a live run on the Task API. Each task is solved, its checked actions are
performed, and it is submitted at once; one stdout line per task (only stdout comes back from the final container).

Nothing may stop the run before every task has a submission: start-up calls after ``POST /runs`` degrade (no status:
assume nothing submitted; no manifest: solve without files), each task catches every error, and at the deadline a
timer submits the fallback for every task still open and finishes the run.
"""

import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from pydantic import Field

from mcnulty.agent.actions import tool_args
from mcnulty.agent.loop import Deadline, Transcripts, solve
from mcnulty.agent.sandbox import SandboxPool, self_test
from mcnulty.agent.submit import FALLBACK, Solution
from mcnulty.clients.arena import (
    MAX_NOTE,
    ArenaClient,
    ArenaError,
    Run,
    RunKind,
    RunManifest,
    RunMeta,
    Submission,
    Task,
    git_meta,
)
from mcnulty.clients.openrouter import BudgetExhausted, CostTracker, OpenRouterClient
from mcnulty.commands.task_filter import TaskFilter
from mcnulty.frozen import FromText
from mcnulty.record.events import CliStarted, DeadlineReached, RunFinished, TaskCrashed, TaskFinished
from mcnulty.record.log import EventLog, Recorder
from mcnulty.settings import Settings

RUN_SECONDS = 30 * 60  # the final container's limit, counted from the process start
DEADLINE_MARGIN = 120.0  # seconds before the hard end at which every task must be submitted


class RunArgs(FromText):
    kind: RunKind
    note: str | None = Field(default=None, max_length=MAX_NOTE)
    seed: int | None = None
    only: TaskFilter = TaskFilter()
    parallel: int = Field(default=12, ge=1)
    collect: bool = False
    max_usd: float = Field(default=2.5, gt=0)  # hard cap on model spend for the whole run (final budget: $3)


def run_deadline(started: float, run: Run):
    """The process start plus the container limit, or the run's own deadline if earlier, minus a margin."""
    end = started + RUN_SECONDS
    if run.deadline_at:
        end = min(end, datetime.fromisoformat(run.deadline_at).timestamp())
    return Deadline(end=end - DEADLINE_MARGIN)


@dataclass(frozen=True, slots=True)
class Submitter:
    """Submits each task once: the worker that solved it, or the deadline rescue, whichever comes first."""

    arena: ArenaClient
    run_id: str
    claimed: set[str] = field(default_factory=set[str])
    lock: threading.Lock = field(default_factory=threading.Lock)

    def open(self, task_id: str):
        with self.lock:
            return task_id not in self.claimed

    def claim(self, task_id: str):
        with self.lock:
            if task_id in self.claimed:
                return False
            self.claimed.add(task_id)
            return True

    def send(self, task_rec: Recorder, task_id: str, submission: Submission):
        try:
            result = self.arena.submit(task_rec, self.run_id, task_id, submission)
        except ArenaError as e:
            task_rec.log(TaskCrashed.from_exception(e))
            if e.status == 409:  # a submission is there already (a restart, or an earlier attempt that arrived)
                return None, ("already submitted",)
            return None, (f"submit failed: {e}",)
        except Exception as e:
            task_rec.log(TaskCrashed.from_exception(e))
            return None, (f"submit failed: {type(e).__name__}: {e}",)
        return result.score, result.feedback


def start_or_resume(rec: Recorder, arena: ArenaClient, args: RunArgs, meta: RunMeta):
    """A new run; for the final after a restart (409: already used) the final run that is still open."""
    try:
        return arena.start_run(rec, args.kind, args.seed if args.kind == "dev" else None, meta)
    except ArenaError as e:
        resumed = arena.active_final(rec) if e.status == 409 and args.kind == "final" else None
        if resumed is None:
            raise
        print(f"resuming final run {resumed.run_id} after a restart")
        return resumed


def run_tasks(settings: Settings, args: RunArgs, started: float):
    run_dir = Path("runs") / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    rec = Recorder(events=EventLog(run_dir=run_dir, secrets=settings.secrets))
    rec.log(CliStarted(argv=tuple(sys.argv[1:])))
    arena = ArenaClient.connect(settings)
    llm = OpenRouterClient.connect(settings, CostTracker(cap=args.max_usd))
    try:
        tools = arena.tools(rec)
        meta = git_meta().model_copy(update={"note": args.note})
        run = start_or_resume(rec, arena, args, meta)
    except ArenaError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    deadline = run_deadline(started, run)
    pool = SandboxPool()
    (run_dir / "run.json").write_text(run.model_dump_json(indent=1), encoding="utf-8")
    (run_dir / "tools.json").write_text("[" + ",".join(t.model_dump_json() for t in tools) + "]", encoding="utf-8")

    # After a container restart the final continues the same run: skip what is already submitted.
    try:
        done = {t.task_id for t in arena.run_status(rec, run.run_id).tasks if t.submitted}
    except ArenaError:
        done = set[str]()  # a second submit gets 409, which is logged
    tasks = [t for t in args.only.select(run.tasks) if t.task_id not in done]
    tasks.sort(key=lambda t: -(t.level or 0))  # the longest tasks first, so none starts in the last wave
    note = f", note: {args.note}" if args.note else ""
    head = f"run {run.run_id} ({run.kind}, theme={run.theme}, seed={run.seed})"
    print(f"{head}, {len(tasks)} tasks, {len(done)} already submitted, dir={run_dir}{note}")
    try:
        manifest = arena.run_manifest(rec, run.run_id)
    except ArenaError:
        manifest = RunManifest()  # every task is solved without files and still submitted
    (run_dir / "manifest.json").write_text(manifest.model_dump_json(indent=1), encoding="utf-8")
    manifest_files = {t.task_id: t.files for t in manifest.tasks}
    if args.collect:
        stats = arena.prefetch(rec, manifest, run.run_id)
        print(f"collect: {sum(x.files for x in stats)} files, {sum(x.failed for x in stats)} failed")
    else:
        print(self_test(rec, pool))

    submitter = Submitter(arena=arena, run_id=run.run_id)
    transcripts = Transcripts()  # images shared by the tasks are read once per run
    finished = threading.Event()

    def finish_run():
        with submitter.lock:
            if finished.is_set():
                return
            finished.set()
        try:
            status = arena.finish(rec, run.run_id)
        except Exception as e:
            rec.log(TaskCrashed.from_exception(e))
            print(f"finish failed: {type(e).__name__}")
            return
        (run_dir / "finish.json").write_text(status.model_dump_json(indent=1), encoding="utf-8")
        summary = {"status", "tasks_submitted", "tasks_total", "accuracy", "cost_usd", "cost_score", "final_score"}
        print(f"finish: {status.model_dump_json(include=summary)}")

    def rescue():
        """At the deadline: the fallback for every task still open, then finish."""
        open_tasks = [t for t in tasks if submitter.claim(t.task_id)]
        rec.log(DeadlineReached(open_tasks=tuple(t.task_id for t in open_tasks)))
        for task in open_tasks:
            task_rec = rec.for_task(task.task_id)
            _, feedback = submitter.send(task_rec, task.task_id, FALLBACK.submission)
            print(f"{task.task_id} L{task.level} deadline: fallback submitted {' | '.join(feedback)}")
        finish_run()

    def solved(task: Task, task_rec: Recorder):
        files = manifest_files.get(task.task_id, {})
        try:
            arena.fetch_task(task_rec, run.run_id, task.task_id, files)
        except Exception as e:
            task_rec.log(TaskCrashed.from_exception(e))
        try:
            return solve(task_rec, llm.for_task(len(tasks)), task, files, arena.cache, pool, deadline, transcripts)
        except BudgetExhausted:
            return FALLBACK
        except Exception as e:
            task_rec.log(TaskCrashed.from_exception(e))
            return FALLBACK

    def perform(task: Task, task_rec: Recorder, solution: Solution):
        for vetted in solution.actions:
            if vetted.blocked:
                continue
            try:
                arena.call_tool(task_rec, run.run_id, task.task_id, vetted.action.tool, tool_args(vetted.action))
            except Exception as e:
                task_rec.log(TaskCrashed.from_exception(e))

    def work(task: Task):
        task_rec = rec.for_task(task.task_id)
        t0 = time.time()
        if not submitter.open(task.task_id):
            return  # the deadline came first
        solution = solved(task, task_rec)
        if not submitter.claim(task.task_id):
            return
        perform(task, task_rec, solution)
        submission = solution.submission
        score, feedback = submitter.send(task_rec, task.task_id, submission)
        result = TaskFinished(
            outcome=submission.outcome,
            seconds=round(time.time() - t0, 1),
            run_cost_usd=llm.tracker.total,
            score=score,
            feedback=feedback,
        )
        task_rec.log(result)
        # one short line per task: only stdout comes back from the final container
        head = f"{task.task_id} L{task.level} {submission.outcome} {result.seconds:.0f}s"
        print(f"{head} ${result.run_cost_usd:.4f} {' | '.join(feedback)}")

    if not args.collect:
        timer = threading.Timer(max(deadline.left(), 0.0), rescue)
        timer.daemon = True
        timer.start()
        with ThreadPoolExecutor(max_workers=args.parallel) as ex:
            list(ex.map(work, tasks))
        timer.cancel()

    finish_run()
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
