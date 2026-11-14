"""The ``mcnulty`` command.

mcnulty run --note "hypothesis" [--tasks t01,t03] [--level 1] [--match regex] [--seed N] [--parallel 12]
mcnulty collect --note "collect"   # download all files, submit nothing, finish: reveals the expected answers
mcnulty check                      # live check of every API (< $0.01)

Every run records all requests and responses to runs/<UTC time>/events.jsonl.
"""

import argparse
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from mcnulty.arena import MAX_NOTE, ArenaClient, ArenaError, RunKind, Task, git_meta
from mcnulty.live_check import live_check
from mcnulty.llm import BudgetExhausted, CostTracker, OpenRouterClient
from mcnulty.recorder import CliStarted, EventLog, Recorder, RunFinished, TaskCrashed, TaskFinished
from mcnulty.settings import Settings, load_settings
from mcnulty.solver import FALLBACK, solve


class RunArgs(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: RunKind
    note: str | None = Field(default=None, max_length=MAX_NOTE)
    seed: int | None = None
    tasks: tuple[str, ...] = ()
    levels: tuple[Literal[1, 2, 3], ...] = ()
    match: re.Pattern[str] | None = None
    limit: int | None = None
    parallel: int = Field(default=12, ge=1)
    collect: bool = False


def build_parser(default_kind: str | None):
    p = argparse.ArgumentParser(prog="mcnulty", description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    commands = p.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="solve and submit tasks, then finish the run")
    run.add_argument("--kind", default=default_kind or "dev", help="dev or final (default: $ARENA_RUN_KIND, else dev)")
    run.add_argument("--tasks", default="", help="comma-separated task ids (default: all)")
    run.add_argument("--level", default="", help="only these levels, e.g. 1 or 1,2")
    run.add_argument("--match", help="only tasks whose instruction matches this regex (case-insensitive)")
    run.add_argument("--limit", type=int, help="solve at most N of the selected tasks")
    run.add_argument("--parallel", type=int, default=12)

    collect = commands.add_parser("collect", help="dev run: download all files, submit nothing, finish")
    for sub in (run, collect):
        sub.add_argument("--note", help=f"hypothesis for this run (max {MAX_NOTE} characters)")
        sub.add_argument("--seed", type=int, help="dev only: replay the tasks of an earlier dev run")

    commands.add_parser("check", help="live check of every API (costs < $0.01)")
    return p


def to_run_args(p: argparse.ArgumentParser, ns: argparse.Namespace):
    if ns.command == "collect":
        data = {"kind": "dev", "note": ns.note, "seed": ns.seed, "collect": True}
    else:
        data = {
            "kind": ns.kind,
            "note": ns.note,
            "seed": ns.seed,
            "tasks": [t for t in ns.tasks.split(",") if t],
            "levels": [int(x) for x in ns.level.split(",") if x],
            "match": f"(?i){ns.match}" if ns.match else None,  # pydantic compiles it; (?i): ignore case
            "limit": ns.limit,
            "parallel": ns.parallel,
        }
    try:
        return RunArgs.model_validate(data)
    except (ValidationError, ValueError) as e:
        p.error(str(e))


def select_tasks(tasks: tuple[Task, ...], args: RunArgs):
    """--tasks, --level, --match and --limit, applied in that order."""
    selected = [t for t in tasks if not args.tasks or t.task_id in args.tasks]
    selected = [t for t in selected if not args.levels or t.level in args.levels]
    selected = [t for t in selected if not args.match or args.match.search(t.instruction)]
    return selected[: args.limit] if args.limit is not None else selected


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
    tasks = [t for t in select_tasks(run.tasks, args) if t.task_id not in done]
    note = f", note: {args.note}" if args.note else ""
    print(
        f"run {run.run_id} ({run.kind}, theme={run.theme}, seed={run.seed}), {len(tasks)} tasks, "
        f"{len(done)} already submitted, dir={run_dir}{note}"
    )
    manifest = arena.run_manifest(rec, run.run_id)
    (run_dir / "manifest.json").write_text(manifest.model_dump_json(indent=1), encoding="utf-8")
    stats = arena.prefetch(rec, manifest, run.run_id)
    print(f"prefetch: {stats.blobs} files, {stats.downloaded} downloaded ({stats.bytes / 1e6:.1f} MB)")

    def work(task: Task):
        task_rec = rec.for_task(task.task_id)
        t0 = time.time()
        try:
            submission = solve(task_rec, arena, llm, run.run_id, task)
        except BudgetExhausted:
            submission = FALLBACK
        except Exception as e:  # one task must never kill the run; always submit something
            task_rec.log(TaskCrashed.from_exception(e))
            submission = FALLBACK
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


def main():
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]
    settings = load_settings()
    p = build_parser(settings.run_kind)
    ns = p.parse_args()
    if ns.command == "check":
        return live_check(settings)
    return run_tasks(settings, to_run_args(p, ns))
