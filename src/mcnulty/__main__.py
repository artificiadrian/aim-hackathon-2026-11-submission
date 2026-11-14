"""CLI: start a run, solve its tasks in parallel, submit each, finish the run, print a summary.

uv run python -m mcnulty --kind dev --note "hypothesis" [--tasks t01,t03] [--level 1] [--parallel 12]
uv run python -m mcnulty --collect --note "collect"   # download all files, submit nothing, finish: reveals answers

Every run records all requests and responses to runs/<UTC time>/events.jsonl.
"""

import argparse
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from mcnulty.arena import MAX_NOTE, ArenaClient, ArenaError, RunKind, Task, git_meta
from mcnulty.llm import BudgetExhausted, CostTracker, OpenRouterClient
from mcnulty.recorder import CliStarted, EventLog, Recorder, TaskCrashed
from mcnulty.settings import load_settings
from mcnulty.solver import FALLBACK, solve


class Args(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: RunKind
    note: str | None = Field(default=None, max_length=MAX_NOTE)
    seed: int | None = None
    tasks: tuple[str, ...] = ()
    levels: tuple[int, ...] = ()
    match: re.Pattern[str] | None = None
    limit: int | None = None
    parallel: int = Field(default=12, ge=1)
    collect: bool = False

    @field_validator("levels")
    @classmethod
    def levels_in_range(cls, levels: tuple[int, ...]):
        if any(level not in (1, 2, 3) for level in levels):
            raise ValueError("levels are 1, 2 or 3")
        return levels


def parse_args(argv: list[str], default_kind: str | None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument(
        "--kind",
        default=default_kind or "dev",
        help="dev or final (default: $ARENA_RUN_KIND, else dev)",
    )
    p.add_argument(
        "--note", help=f"hypothesis for this run (max {MAX_NOTE} characters)"
    )
    p.add_argument(
        "--seed", type=int, help="dev only: replay the tasks of an earlier dev run"
    )
    p.add_argument(
        "--tasks", default="", help="comma-separated task ids (default: all)"
    )
    p.add_argument("--level", default="", help="only these levels, e.g. 1 or 1,2")
    p.add_argument(
        "--match",
        help="only tasks whose instruction matches this regex (case-insensitive)",
    )
    p.add_argument("--limit", type=int, help="solve at most N of the selected tasks")
    p.add_argument("--parallel", type=int, default=12)
    p.add_argument(
        "--collect",
        action="store_true",
        help="dev only: download everything, submit nothing, finish",
    )
    ns = p.parse_args(argv)
    try:
        args = Args(
            kind=ns.kind,
            note=ns.note,
            seed=ns.seed,
            tasks=tuple(t for t in ns.tasks.split(",") if t),
            levels=tuple(int(x) for x in ns.level.split(",") if x),
            match=re.compile(ns.match, re.IGNORECASE) if ns.match else None,
            limit=ns.limit,
            parallel=ns.parallel,
            collect=ns.collect,
        )
    except (ValidationError, ValueError, re.error) as e:
        p.error(str(e))
    if args.collect and args.kind != "dev":
        p.error("--collect is for dev runs only")
    return args


def select_tasks(tasks: tuple[Task, ...], args: Args):
    """--tasks, --level, --match and --limit, applied in that order."""
    selected = [t for t in tasks if not args.tasks or t.task_id in args.tasks]
    selected = [t for t in selected if not args.levels or t.level in args.levels]
    selected = [
        t for t in selected if not args.match or args.match.search(t.instruction)
    ]
    return selected[: args.limit] if args.limit is not None else selected


def main():
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]
    settings = load_settings()
    args = parse_args(sys.argv[1:], settings.run_kind)
    run_dir = Path("runs") / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    rec = Recorder(events=EventLog(run_dir=run_dir, secrets=settings.secrets))
    rec.log(CliStarted(argv=tuple(sys.argv[1:])))

    arena = ArenaClient.connect(settings)
    llm = OpenRouterClient.connect(settings, CostTracker())
    try:
        tools = arena.tools(rec)
        meta = git_meta().model_copy(update={"note": args.note})
        run = arena.start_run(
            rec, args.kind, args.seed if args.kind == "dev" else None, meta
        )
    except ArenaError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    (run_dir / "run.json").write_text(run.model_dump_json(indent=1), encoding="utf-8")
    (run_dir / "tools.json").write_text(
        "[" + ",".join(t.model_dump_json() for t in tools) + "]", encoding="utf-8"
    )

    # After a container restart the final continues the same run: skip what is already submitted.
    done = {t.task_id for t in arena.run_status(rec, run.run_id).tasks if t.submitted}
    tasks = [t for t in select_tasks(run.tasks, args) if t.task_id not in done]
    print(
        f"run {run.run_id} ({run.kind}, theme={run.theme}, seed={run.seed}), {len(tasks)} tasks, "
        f"{len(done)} already submitted, dir={run_dir}"
        + (f", note: {args.note}" if args.note else "")
    )
    manifest = arena.run_manifest(rec, run.run_id)
    (run_dir / "manifest.json").write_text(
        manifest.model_dump_json(indent=1), encoding="utf-8"
    )
    stats = arena.prefetch(rec, manifest, run.run_id)
    print(
        f"prefetch: {stats.blobs} files, {stats.downloaded} downloaded ({stats.bytes / 1e6:.1f} MB)"
    )

    if not args.collect:

        def work(task: Task):
            task_rec = rec.for_task(task.task_id)
            t0 = time.time()
            try:
                submission = solve(task_rec, arena, llm, run.run_id, task)
            except BudgetExhausted:
                submission = FALLBACK
            except Exception as e:  # noqa: BLE001  (one task must never kill the run; always submit something)
                task_rec.log(TaskCrashed(error=f"{type(e).__name__}: {e}"))
                submission = FALLBACK
            try:
                result = " | ".join(
                    arena.submit(
                        task_rec, run.run_id, task.task_id, submission
                    ).feedback
                )
            except ArenaError as e:
                result = f"ERROR {e}"
            # one short line per task: only stdout comes back from the final container
            print(
                f"{task.task_id} L{task.level} {submission.outcome} {time.time() - t0:.0f}s ${llm.tracker.total:.4f} {result}"
            )

        with ThreadPoolExecutor(max_workers=args.parallel) as ex:
            list(ex.map(work, tasks))

    finish = arena.finish(rec, run.run_id)
    (run_dir / "finish.json").write_text(
        finish.model_dump_json(indent=1), encoding="utf-8"
    )
    print(
        f"finish: {finish.model_dump_json(include={'status', 'tasks_submitted', 'tasks_total', 'accuracy', 'cost_usd', 'cost_score', 'final_score'})}"
    )
    t = llm.tracker
    print(
        f"local cost ${t.total:.4f} in {t.calls} calls ({t.prompt_tokens} in / {t.completion_tokens} out)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
