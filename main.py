"""CLI: start a run, solve its tasks in parallel, submit each, finish the run, print a summary.

uv run main.py --kind dev --note "hypothesis" [--tasks t01,t03] [--level 1] [--parallel 12]
uv run main.py --collect --note "data run"     # download all files, submit nothing, finish: reveals expected answers

Every run records all traffic to runs/<UTC time>/events.jsonl (see arena/recorder.py).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from arena import llm, recorder
from arena.client import MAX_NOTE, ArenaClient, ArenaError
from solver import FALLBACK, solve


def select_tasks(
    tasks: list,
    ids: str | None = None,
    levels: str | None = None,
    match: str | None = None,
    limit: int | None = None,
) -> list:
    """--tasks, --level, --match and --limit, applied in that order."""
    if ids:
        want = set(ids.split(","))
        tasks = [t for t in tasks if t.task_id in want]
    if levels:
        want_levels = {int(x) for x in levels.split(",")}
        tasks = [t for t in tasks if t.level in want_levels]
    if match:
        rx = re.compile(match, re.IGNORECASE)
        tasks = [t for t in tasks if rx.search(t.instruction)]
    return tasks[:limit] if limit is not None else tasks


def parse_args() -> argparse.Namespace:
    env_kind = os.environ.get("ARENA_RUN_KIND")  # set in the organisers' container
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument(
        "--kind",
        default=env_kind or "dev",
        help="dev or final (default: $ARENA_RUN_KIND, else dev)",
    )
    p.add_argument(
        "--note", help=f"hypothesis for this run (max {MAX_NOTE} characters)"
    )
    p.add_argument(
        "--seed", type=int, help="dev only: replay the tasks of an earlier dev run"
    )
    p.add_argument("--tasks", help="comma-separated task ids (default: all)")
    p.add_argument("--level", help="only these levels, e.g. 1 or 1,2")
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
    p.add_argument(
        "--yes", action="store_true", help="do not ask for confirmation on --kind final"
    )
    a = p.parse_args()
    if a.kind not in ("dev", "final"):
        p.error(f"ARENA_RUN_KIND must be dev or final, not {a.kind!r}")
    if a.collect and a.kind != "dev":
        p.error("--collect is for dev runs only")
    if a.note and len(a.note) > MAX_NOTE:
        p.error(f"--note is {len(a.note)} characters; the limit is {MAX_NOTE}")
    if a.level and not re.fullmatch(r"[123](,[123])*", a.level):
        p.error(f"--level takes levels 1-3, comma-separated, not {a.level!r}")
    if a.match:
        try:
            re.compile(a.match)
        except re.error as e:
            p.error(f"--match is not a valid regex: {e}")
    return a


def save(path: Path, data: object) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=1, default=str), encoding="utf-8"
    )


def main() -> int:
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]
    a = parse_args()
    out_dir = Path("runs") / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    rec = recorder.start(out_dir)
    rec.emit("cli", argv=sys.argv[1:])

    client = ArenaClient()
    confirm = a.kind == "final" and not a.yes and not os.environ.get("ARENA_RUN_KIND")
    if (
        confirm
        and input(
            "Final runs start from the organisers' runner. Type 'final' to continue: "
        ).strip()
        != "final"
    ):
        return 1
    try:
        tool_specs = client.tools()
        run = client.start_run(
            a.kind,
            seed=a.seed if a.kind == "dev" else None,
            meta={"note": a.note} if a.note else None,
        )
    except ArenaError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    save(out_dir / "run.json", run.model_dump())
    save(out_dir / "tools.json", [t.model_dump() for t in tool_specs])

    tasks = select_tasks(
        run.tasks, ids=a.tasks, levels=a.level, match=a.match, limit=a.limit
    )
    # After a container restart the final continues the same run: skip what is already submitted.
    try:
        done = {
            t["task_id"]
            for t in client.get_run(run.run_id).get("tasks", [])
            if t.get("submitted")
        }
    except ArenaError:
        done = set()
    if done:
        print(f"resuming: {len(done)} task(s) already submitted, skipped")
        tasks = [t for t in tasks if t.task_id not in done]
    seed = (run.model_extra or {}).get("seed")
    print(
        f"run {run.run_id} ({run.kind}, theme={run.theme}{f', seed={seed}' if seed is not None else ''}), "
        f"{len(tasks)} tasks, dir={out_dir}" + (f", note: {a.note}" if a.note else "")
    )

    try:
        st = client.prefetch(run.run_id)
        print(
            f"prefetch: {st['blobs']} files, {st['downloaded']} downloaded ({st['bytes'] / 1e6:.1f} MB)"
        )
    except ArenaError as e:
        print(f"prefetch skipped: {e}")

    tracker = llm.CostTracker()
    results: dict[str, dict] = {}
    if a.collect:
        save(out_dir / "manifest.json", client.run_manifest(run.run_id))
    else:

        def work(task) -> None:
            recorder.set_ctx(task_id=task.task_id)
            t0 = time.time()
            try:
                ans = solve(client, run.run_id, task, tracker)
            except llm.BudgetExhausted:
                ans = FALLBACK
            except Exception as e:  # noqa: BLE001  (one task must never kill the run; always submit something)
                recorder.emit("crash", error=f"{type(e).__name__}: {e}")
                ans = FALLBACK
            try:
                res = client.submit(
                    run.run_id, task.task_id, ans.outcome, ans.answer, ans.sources
                )
            except ArenaError as e:
                res = {"error": str(e)}
            results[task.task_id] = res
            # one short line per task: the final container shows only stdout
            print(
                f"{task.task_id} L{task.level} {ans.outcome} {time.time() - t0:.0f}s ${tracker.total:.4f} {_fmt(res)}"
            )

        with ThreadPoolExecutor(max_workers=max(1, a.parallel)) as ex:
            list(ex.map(work, tasks))

    fin = client.finish(run.run_id)
    save(out_dir / "results.json", results)
    save(out_dir / "finish.json", fin)
    keys = (
        "status",
        "tasks_submitted",
        "tasks_total",
        "accuracy",
        "cost_usd",
        "cost_score",
        "final_score",
        "error",
    )
    print("finish: " + json.dumps({k: fin[k] for k in keys if k in fin}))
    print(
        f"local cost ${tracker.total:.4f} in {tracker.calls} calls ({tracker.prompt_tokens} in / {tracker.completion_tokens} out)"
    )
    rec.close()
    return 0


def _fmt(res: dict) -> str:
    if "error" in res:
        return f"ERROR {res['error'][:200]}"
    fb = res.get("feedback")
    return " | ".join(map(str, fb)) if isinstance(fb, list) else ""


if __name__ == "__main__":
    sys.exit(main())
