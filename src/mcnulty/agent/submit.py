"""The model's tools and its typed final answer: parsing, the locator check and the checked result. Pure functions."""

from typing import Annotated

from pydantic import BeforeValidator, JsonValue, TypeAdapter, ValidationError

from mcnulty.agent.actions import Action, Gate, VettedAction, vet
from mcnulty.agent.citations import fixed_sources
from mcnulty.agent.guards import clean_answer
from mcnulty.clients.arena import Outcome, Source, Submission
from mcnulty.clients.openrouter import FunctionSpec, ToolDefinition
from mcnulty.frozen import JSON, Frozen, ModelInput

PATH_ONLY = (".png", ".jpg", ".jpeg", ".json")  # cited by path alone


def source_from_text(value: object):
    """Models often cite as text: ``"/a.csv:49"`` -> line 49, ``"/a.csv:R-7 (note)"`` -> record R-7."""
    if not isinstance(value, str):
        return value
    path, _, rest = value.partition(":")
    locator = rest.split(" (")[0].strip()
    if not locator:
        return {"path": path.strip()}
    if locator.isdigit():
        return {"path": path.strip(), "line": int(locator)}
    return {"path": path.strip(), "record": locator}


def normal_outcome(value: object):
    return value.strip().upper().replace(" ", "_") if isinstance(value, str) else value


def as_list(value: object):
    """A single source or action where a list belongs becomes a list of one."""
    single: list[object] = [value]
    return single if isinstance(value, str | dict) else value


type ModelOutcome = Annotated[Outcome, BeforeValidator(normal_outcome)]
type ModelSource = Annotated[Source, BeforeValidator(source_from_text)]


class SubmitArgs(ModelInput):
    """What the model submits."""

    outcome: ModelOutcome
    answer: JsonValue = None
    sources: Annotated[tuple[ModelSource, ...], BeforeValidator(as_list)] = ()
    actions: Annotated[tuple[Action, ...], BeforeValidator(as_list)] = ()


class Solution(Frozen):
    """What the solver returns: the submission and the checked actions; the caller performs those not blocked."""

    submission: Submission
    actions: tuple[VettedAction, ...] = ()
    steps: int = 0


FALLBACK = Solution(submission=Submission(outcome="UNSUPPORTED", answer=None))

RUN_PYTHON = ToolDefinition(
    function=FunctionSpec(
        name="run_python",
        description="Run a Python 3.12 script in the sandbox and return what it prints (stdout and stderr).",
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
                "path": {"type": "string", "description": "workspace path of a png or jpg file"},
                "question": {"type": "string", "description": "what to read or look for; ask for all text verbatim"},
            },
            "required": ["path", "question"],
        },
    )
)
STRING: JsonValue = {"type": "string"}
INTEGER: JsonValue = {"type": "integer"}
# Hand-written: the generated schema is 4,600 characters on every call. Pydantic still validates the arguments.
SUBMIT_SCHEMA: dict[str, JsonValue] = {
    "type": "object",
    "properties": {
        "outcome": {"enum": ["OK", "DENIED_SECURITY", "NEEDS_CLARIFICATION", "UNSUPPORTED"]},
        "answer": {"description": "any JSON value, in the format the task asks"},
        "sources": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "path": STRING,
                    "line": INTEGER,
                    "record": STRING,
                    "sheet": STRING,
                    "row": INTEGER,
                    "paragraph": INTEGER,
                    "page": INTEGER,
                },
                "required": ["path"],
            },
        },
        "actions": {
            "type": "array",
            "description": (
                'only what the task asks for: {"tool": "containment", "ops": [{"op": "quarantine_agent", "agent", '
                '"stand_in"?} | {"op": "release_agent", "agent"} | {"op": "revoke_token", "token"} | '
                '{"op": "rotate_token", "token", "at": "YYYY-MM-DDTHH:00:00Z"} | {"op": "block_domain", "domain"} | '
                '{"op": "add_note", "text"}]} | '
                '{"tool": "send_message", "to", "subject", "body"} | '
                '{"tool": "write", "path": "/drafts/...", "content"}'
            ),
            "items": {"type": "object"},
        },
    },
    "required": ["outcome", "answer"],
}
SUBMIT = ToolDefinition(
    function=FunctionSpec(
        name="submit",
        description="Finish the task: the outcome, the answer, its sources and the actions the task asks for.",
        parameters=SUBMIT_SCHEMA,
    )
)
TOOLS = (RUN_PYTHON, VIEW_IMAGE, SUBMIT)


class RunPythonArgs(ModelInput):
    code: str


class ViewImageArgs(ModelInput):
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


class Salvaged(Frozen):
    args: SubmitArgs
    dropped_sources: int
    dropped_actions: tuple[str, ...]  # why each dropped action was invalid


SOURCE: TypeAdapter[Source] = TypeAdapter(ModelSource)
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
    args = SubmitArgs(outcome=raw.outcome, answer=raw.answer, sources=tuple(sources), actions=tuple(actions))
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


def missing_locators_message(missing: list[str]):
    return (
        f"not submitted: these sources have no locator: {', '.join(sorted(set(missing)))}. A file citation earns "
        "nothing; cite each line, record, row, paragraph or page your answer rests on, then call submit again."
    )


class ScriptRun(Frozen):
    """A script's output for the model, and the arguments of a ``submit(...)`` it called, if any."""

    output: str
    submitted: str | None = None


class Accepted(Frozen):
    args: SubmitArgs


class Rejected(Frozen):
    message: str
    asked_locators: bool = False


def check_submit(arguments: str, last: bool, locators_asked: bool, rejected_before: bool) -> Accepted | Rejected:
    """Accept a submission, or say why not. An invalid part is dropped and the rest kept; only a dropped action
    (losing it fails an action task) or a missing locator sends it back, once, and never on the last step."""
    try:
        args = parse_submit(arguments)
    except ValidationError as e:
        salvaged = salvage_submit(arguments)
        if salvaged is None:
            return Rejected(message=f"not submitted, the outcome or the JSON is invalid: {e}")
        if salvaged.dropped_actions and not (last or rejected_before):
            return Rejected(message="not submitted, invalid actions: " + "; ".join(salvaged.dropped_actions))
        args = salvaged.args
    missing = missing_locators(args)
    if missing and not locators_asked and not last:
        return Rejected(message=missing_locators_message(missing), asked_locators=True)
    return Accepted(args=args)


def solution(args: SubmitArgs, gate: Gate, steps: int):
    """The checked result: the answer cleaned, sources only for workspace files, actions only with outcome OK, each
    defanged and vetted."""
    actions = tuple(vet(a, gate) for a in args.actions) if args.outcome == "OK" else ()
    cited = tuple(s for s in args.sources if s.path in gate.paths)
    sources = fixed_sources(cited, gate.records, gate.instruction, gate.paths)
    submission = Submission(outcome=args.outcome, answer=clean_answer(args.answer), sources=sources)
    return Solution(submission=submission, actions=actions, steps=steps)
