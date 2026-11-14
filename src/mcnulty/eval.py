"""Offline evaluation: run the solver on a saved dev run and grade each task against the revealed answers.

The grade follows the platform formula (docs/ARENA_DOCS.md, Scoring): ``answer · (0.7 + 0.3 · sources)`` when the
outcome is right. The platform's partial credit is not published, so this is an approximation:

- scalars match after trimming and ignoring case; numbers match numeric strings ("72" == 72);
- an object scores the mean of its expected keys; a list scores shared items / all items (Jaccard), so order
  is not checked (order tasks score too high when the set is right and the order wrong);
- a source hits when path and locator match, lines within ±2; a transfer cited by another locator does not count;
- a task graded on what the agent did shows the action as its expected answer (``{"quarantine": [...]}``,
  ``{"action": "send_message", "to": ..., "must_mention": [...]}``): the proposed actions that pass the checks are
  turned into that shape (``action_view``); a value counts as mentioned when its text is in the message or draft;
- constraints (the platform's action checks) are not graded.
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pydantic import JsonValue

from mcnulty.actions import BlockDomain, Containment, QuarantineAgent, RotateToken, SendMessage, WriteDraft
from mcnulty.arena import BlobCache, Run, RunManifest, RunStatus, Source, TaskStatus
from mcnulty.llm import BudgetExhausted, OpenRouterClient
from mcnulty.recorder import Frozen, Recorder, RunFinished, TaskCrashed, TaskScored
from mcnulty.solver import FALLBACK, Solution, solve

LINE_TOLERANCE = 2


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


def canonical(value: JsonValue):
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        try:
            return float(text)
        except ValueError:
            return text.casefold()
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def answer_score(got: JsonValue, want: JsonValue) -> float:
    if isinstance(want, dict):
        if not isinstance(got, dict) or not want:
            return float(got == want)
        return sum(answer_score(got.get(key), value) for key, value in want.items()) / len(want)
    if isinstance(want, list):
        if not isinstance(got, list):
            return 0.0
        got_set, want_set = {canonical(v) for v in got}, {canonical(v) for v in want}
        union = got_set | want_set
        return len(got_set & want_set) / len(union) if union else 1.0
    return float(canonical(got) == canonical(want))


def source_hit(got: Source, want: Source):
    if got.path != want.path:
        return False
    if want.line is not None:
        return got.line is not None and abs(got.line - want.line) <= LINE_TOLERANCE
    return (got.page, got.record, got.sheet, got.row, got.paragraph) == (
        want.page,
        want.record,
        want.sheet,
        want.row,
        want.paragraph,
    )


def mentioned(value: JsonValue, text: str):
    items = value if isinstance(value, list) else [value]
    return all(json.dumps(item, ensure_ascii=False).strip('"').casefold() in text.casefold() for item in items)


def action_view(solution: Solution, want: JsonValue):
    """The performed actions in the shape of an action-graded expected answer; else the submitted answer."""
    actions = [v.action for v in solution.actions if not v.blocked]
    if not actions or not isinstance(want, dict):
        return solution.submission.answer
    ops = [op for a in actions if isinstance(a, Containment) for op in a.ops]
    messages = [a for a in actions if isinstance(a, SendMessage)]
    text = "\n".join([m.body for m in messages] + [a.content for a in actions if isinstance(a, WriteDraft)])
    blocked_domains: list[JsonValue] = [*sorted(op.domain for op in ops if isinstance(op, BlockDomain))]
    quarantined: list[JsonValue] = [*sorted(op.agent for op in ops if isinstance(op, QuarantineAgent))]
    rotations: list[JsonValue] = [{"token": op.token, "at": op.at} for op in ops if isinstance(op, RotateToken)]
    view: dict[str, JsonValue] = {
        "quarantine": quarantined,
        "rotate": rotations,
        "block": blocked_domains,
        "domains": blocked_domains,
        "action": "send_message" if messages else "block_domain" if blocked_domains else None,
        "to": messages[0].to if messages else None,
    }
    for key, value in want.items():
        if key not in view:
            view[key] = value if text and mentioned(value, text) else None
    return view


def grade(solution: Solution, expected: TaskStatus):
    submission = solution.submission
    raw = expected.expected_sources if isinstance(expected.expected_sources, list) else []
    wanted = [Source.model_validate(s) for s in raw]
    hits = sum(any(source_hit(got, want) for got in submission.sources) for want in wanted)
    answer = answer_score(action_view(solution, expected.expected_answer), expected.expected_answer)
    right = submission.outcome == expected.expected_outcome
    sources = hits / len(wanted) if wanted else 1.0
    return TaskScored(
        expected_outcome=expected.expected_outcome,
        outcome=submission.outcome,
        answer_score=answer if right else 0.0,
        sources_hit=hits,
        sources_expected=len(wanted),
        score=round(answer * (0.7 + 0.3 * sources), 4) if right else 0.0,
    )


def run_eval(rec: Recorder, llm: OpenRouterClient, blobs: BlobCache, saved: SavedRun, task_ids: tuple[str, ...]):
    """Solve and grade ``task_ids`` of ``saved``; one stdout line per task. Model spend stops at the tracker's cap."""

    def work(task_id: str):
        task = next(t for t in saved.run.tasks if t.task_id == task_id)
        task_rec = rec.for_task(task_id)
        t0 = time.time()
        try:
            solution = solve(task_rec, llm, task, saved.files(task_id), blobs)
        except BudgetExhausted:
            solution = FALLBACK
        except Exception as e:  # one task must never stop the evaluation
            task_rec.log(TaskCrashed.from_exception(e))
            solution = FALLBACK
        scored = grade(solution, saved.expected(task_id))
        task_rec.log(scored)
        print(
            f"{task_id} L{task.level} {scored.outcome}/{scored.expected_outcome} answer={scored.answer_score:.2f} "
            f"sources={scored.sources_hit}/{scored.sources_expected} score={scored.score:.2f} "
            f"{time.time() - t0:.0f}s ${llm.tracker.total:.4f}"
        )
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
