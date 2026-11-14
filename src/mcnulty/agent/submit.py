"""The model's tools and its typed final answer: parsing, the locator check and the checked result. Pure functions."""

from typing import Annotated

from pydantic import BeforeValidator, JsonValue, ValidationError

from mcnulty.agent.actions import Action, VettedAction, vet
from mcnulty.clients.arena import Outcome, Source, Submission
from mcnulty.clients.openrouter import FunctionSpec, ToolDefinition
from mcnulty.frozen import JSON, Frozen

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


class SubmitArgs(Frozen):
    outcome: Outcome
    answer: JsonValue = None
    sources: tuple[Annotated[Source, BeforeValidator(source_from_text)], ...] = ()
    actions: tuple[Action, ...] = ()


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
SUBMIT = ToolDefinition(
    function=FunctionSpec(
        name="submit",
        description="Finish the task: the outcome, the answer, its sources and the actions the task asks for.",
        parameters=SubmitArgs.model_json_schema(),
    )
)
TOOLS = (RUN_PYTHON, VIEW_IMAGE, SUBMIT)


class RunPythonArgs(Frozen):
    code: str


class ViewImageArgs(Frozen):
    path: str
    question: str


def parse_submit(arguments: str):
    """``SubmitArgs`` from the model's arguments; an answer given as a JSON string becomes its value."""
    args = SubmitArgs.model_validate_json(arguments)
    if isinstance(args.answer, str) and args.answer.strip()[:1] in {"[", "{"}:
        try:
            value = JSON.validate_json(args.answer)
        except ValidationError:
            return args
        return args.model_copy(update={"answer": value})
    return args


class OutcomeAndAnswer(Frozen, extra="ignore"):
    outcome: Outcome
    answer: JsonValue = None


def salvage_submit(arguments: str):
    """The last step's submit when it does not validate: keep outcome and answer, drop sources and actions."""
    try:
        kept = OutcomeAndAnswer.model_validate_json(arguments)
    except ValidationError:
        return None
    return parse_submit(kept.model_dump_json())


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


def solution(args: SubmitArgs, verified: frozenset[str], rules: tuple[str, ...], steps: int):
    """The checked result: actions only with outcome OK, each defanged and vetted."""
    actions = tuple(vet(a, verified, rules) for a in args.actions) if args.outcome == "OK" else ()
    submission = Submission(outcome=args.outcome, answer=args.answer, sources=args.sources)
    return Solution(submission=submission, actions=actions, steps=steps)
