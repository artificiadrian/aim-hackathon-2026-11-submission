"""The model's tools and its typed final answer: parsing, the locator check and the checked result. Pure functions."""

import json
from typing import Annotated

from pydantic import BeforeValidator, JsonValue, TypeAdapter, ValidationError

from mcnulty.agent.actions import Action, Gate, VettedAction, vet
from mcnulty.agent.citations import fixed_sources
from mcnulty.agent.guards import clean_answer, redact_tokens
from mcnulty.clients.arena import Outcome, Source, Submission
from mcnulty.clients.openrouter import FunctionSpec, ToolDefinition
from mcnulty.frozen import JSON, Frozen, ModelInput

PATH_ONLY = (".png", ".jpg", ".jpeg", ".json")  # cited by path alone


def normal_outcome(value: object):
    return value.strip().upper().replace(" ", "_") if isinstance(value, str) else value


def as_list(value: object):
    """A single source or action where a list belongs becomes a list of one; nested lists are flattened one level
    (models write ``sources=[cite(...), cite(...)]``, and ``cite`` returns a list)."""
    if isinstance(value, str | dict):
        single: list[object] = [value]
        return single
    if isinstance(value, list | tuple):
        items: list[object] = []
        for item in value:  # pyright: ignore[reportUnknownVariableType] - any model JSON
            items.extend(item if isinstance(item, list) else [item])  # pyright: ignore[reportUnknownArgumentType]
        return items
    return value


type ModelOutcome = Annotated[Outcome, BeforeValidator(normal_outcome)]


class SubmitArgs(ModelInput):
    """What the model submits."""

    outcome: ModelOutcome
    answer: JsonValue = None
    sources: Annotated[tuple[Source, ...], BeforeValidator(as_list)] = ()
    actions: Annotated[tuple[Action, ...], BeforeValidator(as_list)] = ()
    notes: tuple[str, ...] = ()  # citation problems found by submit(...) in the sandbox (ws.submit)


class Solution(Frozen):
    """What the solver returns: the submission and the checked actions; the caller performs those not blocked."""

    submission: Submission
    actions: tuple[VettedAction, ...] = ()
    steps: int = 0


FALLBACK = Solution(submission=Submission(outcome="UNSUPPORTED", answer=None))

RUN_PYTHON = ToolDefinition(
    function=FunctionSpec(
        name="run_python",
        description=(
            "Run a Python 3.12 cell in the task's session (variables stay); returns what it prints. submit(...) in "
            "it finishes."
        ),
        parameters={"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]},
    )
)
VIEW_IMAGE = ToolDefinition(
    function=FunctionSpec(
        name="view_image",
        description="Look at an image file of the workspace and answer a question about it (text, tables, details).",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "workspace path of a png or jpg file; cite the image by it"},
                "question": {"type": "string", "description": "what to read or look for; ask for all text verbatim"},
            },
            "required": ["path", "question"],
        },
    )
)
CLASSIFY = ToolDefinition(
    function=FunctionSpec(
        name="classify",
        description=(
            "Ask one yes/no question about every line of a text file (jsonl, csv, md, txt), e.g. which board posts "
            "are injected. A fast judge answers per line; you get the likely-yes lines and the unsure ones, each with "
            "its citation. Put the full definition from the rules into the question. Use it for free text; compute "
            "facts from structured fields with code."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "workspace path of a jsonl, csv, md or txt file"},
                "question": {"type": "string", "description": "a yes/no question about one line, with the definition"},
            },
            "required": ["path", "question"],
        },
    )
)
LOCATED_SOURCE: dict[str, JsonValue] = {
    "type": "object",
    "properties": {
        "path": {"type": "string"},
        "line": {"type": "integer"},
        "row": {"type": "integer"},
        "paragraph": {"type": "integer"},
        "page": {"type": "integer"},
        "record": {"type": "string"},
        "sheet": {"type": "string"},
    },
    "required": ["path"],
}
SUBMIT = ToolDefinition(
    function=FunctionSpec(
        name="submit",
        description="Submit the final answer with the values written out (only when a script cannot submit).",
        parameters={
            "type": "object",
            "properties": {
                "outcome": {"type": "string", "enum": ["OK", "DENIED_SECURITY", "NEEDS_CLARIFICATION", "UNSUPPORTED"]},
                "answer": {"description": "any JSON value, in the format the task asks"},
                "sources": {"type": "array", "items": LOCATED_SOURCE},
                "actions": {"type": "array", "items": {"type": "object"}},
            },
            "required": ["outcome", "answer"],
        },
    )
)
TOOLS = (RUN_PYTHON, VIEW_IMAGE, CLASSIFY)  # finishing is submit(...) inside a script (packages/sandbox: ws.submit)


class RunPythonArgs(ModelInput):
    code: str


class ViewImageArgs(ModelInput):
    path: str
    question: str


class ClassifyArgs(ModelInput):
    path: str
    question: str


def with_answer_value(args: SubmitArgs):
    """An answer given as JSON text (a list or object written as a string) becomes its value."""
    if isinstance(args.answer, str) and args.answer.strip()[:1] in {"[", "{"}:
        try:
            return args.model_copy(update={"answer": JSON.validate_json(args.answer)})
        except ValidationError:
            return args
    return args


def parse_submit(arguments: str):
    """``SubmitArgs`` from the model's arguments; raises ``ValidationError`` when any part is invalid."""
    return with_answer_value(SubmitArgs.model_validate_json(arguments))


class RawSubmit(ModelInput):
    outcome: ModelOutcome
    answer: JsonValue = None
    sources: Annotated[tuple[JsonValue, ...], BeforeValidator(as_list)] = ()
    actions: Annotated[tuple[JsonValue, ...], BeforeValidator(as_list)] = ()
    notes: tuple[str, ...] = ()


class Salvaged(Frozen):
    args: SubmitArgs
    dropped_sources: int
    dropped_actions: tuple[str, ...]  # why each dropped action was invalid


SOURCE: TypeAdapter[Source] = TypeAdapter(Source)
ACTION: TypeAdapter[Action] = TypeAdapter(Action)


def salvage_submit(arguments: str):
    """Keep every part of an invalid submission that is valid on its own: outcome, answer, each source and action.
    None when even the outcome is unusable."""
    try:
        raw = RawSubmit.model_validate_json(arguments)
    except ValidationError:
        return None
    sources: list[Source] = []
    for item in raw.sources:
        try:
            sources.append(SOURCE.validate_python(item))
        except ValidationError:
            continue
    actions: list[Action] = []
    dropped: list[str] = []
    for item in raw.actions:
        try:
            actions.append(ACTION.validate_python(item))
        except ValidationError as e:
            dropped.append(f"{item}: {e}")
    args = SubmitArgs(
        outcome=raw.outcome, answer=raw.answer, sources=tuple(sources), actions=tuple(actions), notes=raw.notes
    )
    return Salvaged(
        args=with_answer_value(args),
        dropped_sources=len(raw.sources) - len(sources),
        dropped_actions=tuple(dropped),
    )


def missing_locators(args: SubmitArgs):
    """Sources that name only a file whose type needs a line, record, row, paragraph or page."""
    return [
        s.path
        for s in args.sources
        if not s.path.lower().endswith(PATH_ONLY) and s.locators == (None,) * len(s.locators)
    ]


REDACTION_NOTE = (
    "the answer holds a secret token value, which the harness replaces with [redacted]: name the token by its id "
    "or name as the rules and the task use it, not by its value"
)


def missing_locators_message(missing: list[str]):
    return (
        f"these sources have no locator: {', '.join(sorted(set(missing)))}; a file citation earns nothing: cite "
        'each line, record, row, paragraph or page your answer rests on, e.g. "/fleet/x.csv#record=ID" or '
        '"/logs/x.jsonl#line=N"'
    )


class ScriptRun(Frozen):
    """A script's output for the model, and the arguments of a ``submit(...)`` it called, if any."""

    output: str
    submitted: str | None = None


class Accepted(Frozen):
    args: SubmitArgs


class Rejected(Frozen):
    message: str
    asked_citations: bool = False
    args: SubmitArgs | None = None  # the parsed submission, kept as the best answer so far


def check_submit(arguments: str, final: bool, citations_asked: bool, rejected_before: bool) -> Accepted | Rejected:
    """Accept a submission, or say why not. An invalid part is dropped and the rest kept; only a dropped action
    (losing it fails an action task) or a problem with the citations (no locator; a note from the sandbox's check
    against the files) sends it back, once each, and never when ``final`` (no call is left to fix it)."""
    try:
        args = parse_submit(arguments)
    except ValidationError as e:
        salvaged = salvage_submit(arguments)
        if salvaged is None:
            return Rejected(message=f"not submitted, the outcome or the JSON is invalid: {e}")
        if salvaged.dropped_actions and not (final or rejected_before):
            reason = "not submitted, invalid actions: " + "; ".join(salvaged.dropped_actions)
            return Rejected(message=reason, args=salvaged.args)
        args = salvaged.args
    missing = missing_locators(args)
    problems = [*args.notes, *([missing_locators_message(missing)] if missing else [])]
    if redact_tokens(json.dumps(args.answer)) != json.dumps(args.answer):
        problems.append(REDACTION_NOTE)  # eval A t25: redacted silently, the model never learned why
    if problems and not citations_asked and not final:
        listed = "\n".join(f"- {p}" for p in problems)
        message = (
            f"not submitted yet; the citations need a look:\n{listed}\nFix them from your session's variables "
            '(sources=list(rows["_source"]), hits, cite(path, value)) and submit again; submitting the same again '
            "keeps them."
        )
        return Rejected(message=message, asked_citations=True, args=args)
    return Accepted(args=args)


def blocked_message(solved: Solution):
    """Why the guards block actions of a submission, for the model to repair once; empty when none is blocked."""
    lines = [f"- {f.kind}: {f.detail}" for v in solved.actions if v.blocked for f in v.findings if f.blocks]
    if not lines:
        return ""
    listed = "\n".join(lines)
    return (
        f"not submitted yet; the harness would not perform these actions:\n{listed}\nFix the actions (or leave out "
        "those the task does not ask for) and submit again; submitting the same again drops them."
    )


def solution(args: SubmitArgs, gate: Gate, steps: int):
    """The checked result: the answer cleaned, sources only for workspace files, actions only with outcome OK, each
    defanged and vetted."""
    actions = tuple(vet(a, gate) for a in args.actions) if args.outcome == "OK" else ()
    cited = tuple(s for s in args.sources if s.path in gate.paths)
    sources = fixed_sources(cited, gate.records)
    submission = Submission(outcome=args.outcome, answer=clean_answer(args.answer), sources=sources)
    return Solution(submission=submission, actions=actions, steps=steps)
