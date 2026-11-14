"""Offline grade of one solution against a dev run's revealed answers. Pure functions.

The grade follows the platform formula (docs/ARENA_DOCS.md, Scoring): ``answer · (0.7 + 0.3 · sources)`` when the
outcome is right. The platform's partial credit is not published, so this is an approximation:

- scalars match after trimming and ignoring case; numbers match numeric strings ("72" == 72);
- booleans match only booleans;
- an object scores the mean of its expected keys; a sorted list scores shared items / all items (Jaccard); an
  expected list of scalars that is not sorted is an order (a timeline, a chain) and scores only when equal; a list of
  objects is a set;
- a source hits when path and locator match, lines within ±2; a transfer cited by another locator does not count;
- a task graded on what the agent did shows the action as its expected answer (``{"quarantine": [...]}``,
  ``{"action": "send_message", "to": ..., "must_mention": [...]}``): the proposed actions that pass the checks are
  turned into that shape (``action_view``); each item of a value counts as mentioned when its text (numbers also
  with thousands separators) is in the message or draft;
- constraints (the platform's action checks) are not graded.
"""

import json

from pydantic import JsonValue

from mcnulty.agent.actions import BlockDomain, Containment, QuarantineAgent, RotateToken, SendMessage, WriteDraft
from mcnulty.agent.submit import Solution
from mcnulty.clients.arena import Source, TaskStatus
from mcnulty.record.events import TaskScored

LINE_TOLERANCE = 2


def canonical(value: JsonValue):
    if isinstance(value, bool):
        return f"bool:{value}"  # True == 1.0 in Python
    if value is None:
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


UNORDERED_KEYS = frozenset({"must_mention"})  # a set of things a message must name, whatever their order


def answer_score(got: JsonValue, want: JsonValue, ordered: bool = True) -> float:
    if isinstance(want, dict):
        if not isinstance(got, dict) or not want:
            return float(got == want)
        scores = (answer_score(got.get(key), value, key not in UNORDERED_KEYS) for key, value in want.items())
        return sum(scores) / len(want)
    if isinstance(want, list):
        if not isinstance(got, list):
            return 0.0
        scalars = not any(isinstance(v, dict | list) for v in want)  # a list of objects is a set (rotations, ops)
        if ordered and scalars and want != sorted(want, key=lambda v: str(canonical(v))):  # a timeline, a chain
            return float([canonical(v) for v in got] == [canonical(v) for v in want])
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


def forms(item: JsonValue):
    plain = json.dumps(item, ensure_ascii=False).strip('"')
    return (plain, f"{item:,}") if isinstance(item, int) and not isinstance(item, bool) else (plain,)


def mentioned(value: JsonValue, text: str):
    """The items of ``value`` whose text is in ``text`` (a list keeps its found items, a scalar itself or None)."""
    folded = text.casefold()
    found = [
        v for v in (value if isinstance(value, list) else [value]) if any(f.casefold() in folded for f in forms(v))
    ]
    return found if isinstance(value, list) else (value if found else None)


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
    kinds = [op.op for op in ops]
    rotations: list[JsonValue] = [{"token": op.token, "at": op.at} for op in ops if isinstance(op, RotateToken)]
    view: dict[str, JsonValue] = {
        "quarantine": quarantined,
        "rotate": rotations,
        "block": blocked_domains,
        "domains": blocked_domains,
        "action": "send_message" if messages else kinds[0] if len(set(kinds)) == 1 else None,
        "agent": quarantined[0] if len(quarantined) == 1 else None,
        "to": messages[0].to if messages else None,
    }
    for key, value in want.items():
        if key not in view:
            view[key] = mentioned(value, text) if text else None
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
