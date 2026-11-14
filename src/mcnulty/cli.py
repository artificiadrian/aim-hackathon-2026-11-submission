"""The ``mcnulty`` command.

mcnulty run --note "hypothesis" [--tasks t01,t03] [--level 1] [--match regex] [--seed N] [--parallel 12]
mcnulty collect --note "collect"   # download all files, submit nothing, finish: reveals the expected answers
mcnulty eval runs/<dir> --max-usd 0.10 [--tasks t01,t03]   # offline: solve a saved dev run, grade locally
mcnulty check                      # live check of every API (< $0.01)

Every run records all requests and responses to runs/<UTC time>/events.jsonl.
"""

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from mcnulty.actions import tool_args
from mcnulty.arena import MAX_NOTE, ArenaClient, ArenaError, BlobCache, RunKind, Task, git_meta
from mcnulty.eval import SavedRun, run_eval
from mcnulty.live_check import live_check
from mcnulty.llm import BudgetExhausted, CostTracker, OpenRouterClient
from mcnulty.recorder import CliStarted, EventLog, Recorder, RunFinished, TaskCrashed, TaskFinished
from mcnulty.settings import Settings, load_settings
from mcnulty.solver import FALLBACK, solve


class TaskFilter(BaseModel):
    """--tasks, --level, --match and --limit, applied in that order."""

    model_config = ConfigDict(frozen=True)

    tasks: tuple[str, ...] = ()
    levels: tuple[Literal[1, 2, 3], ...] = ()
    match: re.Pattern[str] | None = None
    limit: int | None = None

    def select(self, tasks: tuple[Task, ...]):
        selected = [t for t in tasks if not self.tasks or t.task_id in self.tasks]
        selected = [t for t in selected if not self.levels or t.level in self.levels]
        selected = [t for t in selected if not self.match or self.match.search(t.instruction)]
        return selected[: self.limit] if self.limit is not None else selected


class RunArgs(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: RunKind
    note: str | None = Field(default=None, max_length=MAX_NOTE)
    seed: int | None = None
    only: TaskFilter = TaskFilter()
    parallel: int = Field(default=12, ge=1)
    collect: bool = False


class EvalArgs(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_dir: Path
    note: str | None = Field(default=None, max_length=MAX_NOTE)
    only: TaskFilter = TaskFilter()
    max_usd: float = Field(gt=0, le=2.0)


def build_parser(default_kind: str | None):
    p = argparse.ArgumentParser(prog="mcnulty", description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    commands = p.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="solve and submit tasks, then finish the run")
    run.add_argument("--kind", default=default_kind or "dev", help="dev or final (default: $ARENA_RUN_KIND, else dev)")
    run.add_argument("--parallel", type=int, default=12)

    evaluate = commands.add_parser("eval", help="offline: solve a saved dev run and grade it against its answers")
    evaluate.add_argument("run_dir", type=Path, help="a finished dev run in runs/ (with finish.json)")
    evaluate.add_argument("--max-usd", type=float, required=True, help="hard spend cap for model calls (at most 2)")
    evaluate.add_argument("--note", help="hypothesis for this evaluation")

    for sub in (run, evaluate):
        sub.add_argument("--tasks", default="", help="comma-separated task ids (default: all)")
        sub.add_argument("--level", default="", help="only these levels, e.g. 1 or 1,2")
        sub.add_argument("--match", help="only tasks whose instruction matches this regex (case-insensitive)")
        sub.add_argument("--limit", type=int, help="at most N of the selected tasks")

    collect = commands.add_parser("collect", help="dev run: download all files, submit nothing, finish")
    for sub in (run, collect):
        sub.add_argument("--note", help=f"hypothesis for this run (max {MAX_NOTE} characters)")
        sub.add_argument("--seed", type=int, help="dev only: replay the tasks of an earlier dev run")

    commands.add_parser("check", help="live check of every API (costs < $0.01)")
    return p


def filter_data(ns: argparse.Namespace):
    return {
        "tasks": [t for t in ns.tasks.split(",") if t],
        "levels": [int(x) for x in ns.level.split(",") if x],
        "match": f"(?i){ns.match}" if ns.match else None,  # pydantic compiles it; (?i): ignore case
        "limit": ns.limit,
    }


def to_args(p: argparse.ArgumentParser, ns: argparse.Namespace):
    try:
        if ns.command == "collect":
            return RunArgs.model_validate({"kind": "dev", "note": ns.note, "seed": ns.seed, "collect": True})
        if ns.command == "eval":
            data = {"run_dir": ns.run_dir, "note": ns.note, "only": filter_data(ns), "max_usd": ns.max_usd}
            return EvalArgs.model_validate(data)
        data = {"kind": ns.kind, "note": ns.note, "seed": ns.seed, "only": filter_data(ns), "parallel": ns.parallel}
        return RunArgs.model_validate(data)
    except (ValidationError, ValueError) as e:
        p.error(str(e))


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


def main():
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]
    settings = load_settings()
    p = build_parser(settings.run_kind)
    ns = p.parse_args()
    if ns.command == "check":
        return live_check(settings)
    args = to_args(p, ns)
    if isinstance(args, EvalArgs):
        return eval_tasks(settings, args)
    return run_tasks(settings, args)
