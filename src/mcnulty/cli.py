"""The ``mcnulty`` command.

mcnulty run --note "hypothesis" [--tasks t01,t03] [--level 1] [--match regex] [--seed N] [--parallel 12]
mcnulty collect --note "collect"   # download all files, submit nothing, finish: reveals the expected answers
mcnulty eval runs/<dir> --max-usd 0.10 [--tasks t01,t03]   # offline: solve a saved dev run, grade locally
mcnulty check                      # live check of every API (< $0.01)

Every run records all requests and responses to runs/<UTC time>/events.jsonl.
"""

import argparse
import io
import sys
import time
from pathlib import Path

from pydantic import ValidationError

from mcnulty.clients.arena import MAX_NOTE
from mcnulty.commands.check import live_check
from mcnulty.commands.evaluate import EvalArgs, eval_tasks
from mcnulty.commands.run import RunArgs, run_tasks
from mcnulty.settings import load_settings


class Namespace(argparse.Namespace):
    """The parsed command line, typed; ``to_args`` validates it into the command's model."""

    command: str = ""
    kind: str = "dev"
    parallel: int = 12
    run_dir: Path = Path()
    max_usd: float = 0.0
    note: str | None = None
    seed: int | None = None
    tasks: str = ""
    level: str = ""
    match: str | None = None
    limit: int | None = None


def build_parser(default_kind: str | None):
    p = argparse.ArgumentParser(prog="mcnulty", description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    commands = p.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="solve and submit tasks, then finish the run")
    run.add_argument("--kind", default=default_kind or "dev", help="dev or final (default: $ARENA_RUN_KIND, else dev)")
    run.add_argument("--parallel", type=int, default=12)
    run.add_argument("--max-usd", type=float, default=1.5, help="hard spend cap for model calls (final budget: $3)")

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


def filter_data(ns: Namespace):
    return {
        "tasks": [t for t in ns.tasks.split(",") if t],
        "levels": [int(x) for x in ns.level.split(",") if x],
        "match": f"(?i){ns.match}" if ns.match else None,  # pydantic compiles it; (?i): ignore case
        "limit": ns.limit,
    }


def to_args(p: argparse.ArgumentParser, ns: Namespace):
    try:
        if ns.command == "collect":
            return RunArgs.model_validate({"kind": "dev", "note": ns.note, "seed": ns.seed, "collect": True})
        if ns.command == "eval":
            data = {"run_dir": ns.run_dir, "note": ns.note, "only": filter_data(ns), "max_usd": ns.max_usd}
            return EvalArgs.model_validate(data)
        data = {
            "kind": ns.kind,
            "note": ns.note,
            "seed": ns.seed,
            "only": filter_data(ns),
            "parallel": ns.parallel,
            "max_usd": ns.max_usd,
        }
        return RunArgs.model_validate(data)
    except (ValidationError, ValueError) as e:
        p.error(str(e))


def main():
    started = time.time()  # the final's 30 minutes count from the container start
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(line_buffering=True)
    settings = load_settings()
    p = build_parser(settings.run_kind)
    ns = p.parse_args(namespace=Namespace())
    if ns.command == "check":
        return live_check(settings)
    args = to_args(p, ns)
    if isinstance(args, EvalArgs):
        return eval_tasks(settings, args)
    return run_tasks(settings, args, started)
