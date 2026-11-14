"""The agent loop: one model works on the task's real files by writing Python, then calls a typed ``submit``.

The solver sees only the task and its files (path -> sha256 in the blob cache), never the Task API, so the same code
runs live (``mcnulty run``) and offline on saved runs (``mcnulty eval``). Actions come back as data, already checked;
the caller performs them. No task knowledge: the workspace's own operator files hold the rules. This module holds the
side effects (blob reads, model calls, sandbox runs); ``prompt`` and ``submit`` are pure.
"""

import base64
import math
import re
import time

from pydantic import ValidationError

from mcnulty.agent.actions import Gate, verified_contacts
from mcnulty.agent.citations import line_records
from mcnulty.agent.guards import action_scope
from mcnulty.agent.prompt import (
    CUT_REPLY,
    LAST_STEP,
    SYSTEM,
    USE_A_TOOL,
    first_message,
    script_output,
    shorten_old_outputs,
)
from mcnulty.agent.sandbox import TaskSandbox
from mcnulty.agent.submit import (
    FALLBACK,
    SUBMIT,
    TOOLS,
    Accepted,
    Rejected,
    RunPythonArgs,
    ScriptRun,
    Solution,
    ViewImageArgs,
    check_submit,
    solution,
)
from mcnulty.blobs import BlobCache
from mcnulty.clients.arena import Task
from mcnulty.clients.openrouter import (
    AssistantMessage,
    BudgetExhausted,
    LLMError,
    Message,
    OpenRouterClient,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from mcnulty.frozen import Frozen
from mcnulty.record.events import ErrorInfo, TaskSolved, ToolFailed
from mcnulty.record.log import Recorder

MAX_STEPS = 12  # model calls per task; the last one must submit
MAX_TOKENS = 3000  # per model reply
LAST_CALL_LEFT = 240.0  # seconds before the deadline: the next call is the submit-only last step
CALL_RESERVE = 60.0  # seconds before the deadline at which a model call stops waiting or retrying
OPERATOR_FILES = ("/AGENTS.md", "/runbook.md")
CLOSED = re.compile(r"(?mi)^status:\s*(?:closed|resolved|done)\s*$")  # in a ticket's front matter
IMAGE_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}


class Deadline(Frozen):
    """When every task must be submitted (epoch seconds). The loops hurry before it; ``mcnulty run`` submits the
    fallback for every open task at it."""

    end: float

    def left(self):
        return self.end - time.time()


NO_DEADLINE = Deadline(end=math.inf)


class TaskFiles(Frozen):
    """A task's files as the loop needs them, read once from the blob cache."""

    paths: dict[str, str]  # path -> sha256
    sizes: dict[str, int]
    trusted: dict[str, str]  # operator files and open tickets: path -> text
    history: dict[str, str]  # closed tickets: shown as history, never as a request
    contacts: str  # /contacts.csv, empty when missing
    records: dict[str, dict[int, str]]  # csv path -> line -> record id (``citations.line_records``)

    @classmethod
    def load(cls, files: dict[str, str], blobs: BlobCache):
        def text(path: str):
            data = blobs.get(files[path]) if path in files else None
            return data.decode("utf-8", errors="replace") if data is not None else ""

        tickets = {p: text(p) for p in sorted(files) if p.startswith("/tickets/")}
        operator = {p: text(p) for p in OPERATOR_FILES if p in files}
        return cls(
            paths=files,
            sizes={p: len(blobs.get(sha) or b"") for p, sha in files.items()},
            trusted=operator | {p: t for p, t in tickets.items() if not CLOSED.search(t)},
            history={p: t for p, t in tickets.items() if CLOSED.search(t)},
            contacts=text("/contacts.csv"),
            records={p: line_records(text(p)) for p in files if p.endswith(".csv")},
        )

    @property
    def rules(self):
        return tuple(self.trusted[p] for p in OPERATOR_FILES if p in self.trusted)

    @property
    def requests(self):
        """The open tickets' text: with the instruction, the only text that may ask for an action."""
        return "\n".join(t for p, t in self.trusted.items() if p not in OPERATOR_FILES)


class Tools(Frozen):
    """What the model's tools act on."""

    llm: OpenRouterClient
    box: TaskSandbox
    files: TaskFiles
    blobs: BlobCache

    def run_python(self, rec: Recorder, args: RunPythonArgs):
        run = self.box.run(rec, args.code)
        return ScriptRun(output=script_output(run), submitted=run.submitted)

    def view_image(self, rec: Recorder, args: ViewImageArgs):
        suffix = "." + args.path.rsplit(".", 1)[-1].lower()
        data = self.blobs.get(self.files.paths[args.path]) if args.path in self.files.paths else None
        if data is None or suffix not in IMAGE_TYPES:
            return f"error: {args.path} is not an image file of the workspace (use the full path, e.g. /scans/a.jpg)"
        url = f"data:{IMAGE_TYPES[suffix]};base64,{base64.b64encode(data).decode()}"
        return self.llm.describe_image(rec, url, args.question, max_tokens=1500)

    def call(self, rec: Recorder, call: ToolCall):
        """Run one tool call. ``submit`` comes back as its arguments; a bad call or a failed tool becomes an error
        text for the model, so the task goes on. Only the spend cap ends it."""
        name, arguments = call.function.name, call.function.arguments
        try:
            if name == "submit":
                return ScriptRun(output="", submitted=arguments)
            if name == "run_python":
                return self.run_python(rec, RunPythonArgs.model_validate_json(arguments))
            if name == "view_image":
                return ScriptRun(output=self.view_image(rec, ViewImageArgs.model_validate_json(arguments)))
        except ValidationError as e:
            return ScriptRun(output=f"error in the arguments of {name}: {e}")
        except BudgetExhausted:
            raise
        except (OSError, LLMError) as e:
            rec.log(ToolFailed(tool=name, error=ErrorInfo.from_exception(e)))
            return ScriptRun(output=f"error: {name} failed ({type(e).__name__}); try again or work without it")
        return ScriptRun(output=f"error: no tool named {name}")


def solve(
    rec: Recorder,
    llm: OpenRouterClient,
    task: Task,
    files: dict[str, str],
    blobs: BlobCache,
    deadline: Deadline = NO_DEADLINE,
):
    task_files = TaskFiles.load(files, blobs)
    box = TaskSandbox.create(rec, task.task_id, files, blobs)
    try:
        result = loop(rec, Tools(llm=llm, box=box, files=task_files, blobs=blobs), task, deadline)
    finally:
        box.remove()
    rec.log(TaskSolved(solution=result.model_dump(mode="json")))
    return result


def loop(rec: Recorder, tools: Tools, task: Task, deadline: Deadline) -> Solution:  # too complex for return inference
    files = tools.files
    gate = Gate(
        paths=frozenset(files.paths),
        verified=verified_contacts(files.contacts),
        rules=files.rules,
        scope=action_scope(f"{task.instruction}\n{files.requests}"),
        instruction=task.instruction,
        records=files.records,
    )
    messages: list[Message] = [
        SystemMessage(content=SYSTEM),
        UserMessage(content=first_message(task.instruction, files.trusted, files.history, files.sizes)),
    ]
    locators_asked = False
    rejected_before = False
    for step in range(1, MAX_STEPS + 1):
        last = step == MAX_STEPS or deadline.left() < LAST_CALL_LEFT
        if last:
            messages.append(UserMessage(content=LAST_STEP))
        reply = tools.llm.chat(
            rec,
            shorten_old_outputs(tuple(messages)),
            tools=(SUBMIT,) if last else TOOLS,
            max_tokens=MAX_TOKENS,
            reasoning_effort="none",
            require_tool=last,
            budget=max(deadline.left() - CALL_RESERVE, 10.0),
        )
        choice = reply.choices[0]
        message = choice.message
        messages.append(AssistantMessage(content=message.content, tool_calls=message.tool_calls))
        if not message.tool_calls:
            messages.append(UserMessage(content=USE_A_TOOL))
            continue
        if choice.finish_reason == "length" and not last:  # the arguments may be cut: run none of them
            messages += [ToolMessage(tool_call_id=call.id, content=CUT_REPLY) for call in message.tool_calls]
            continue
        for call in message.tool_calls:
            run = tools.call(rec, call)
            if run.submitted is None:
                messages.append(ToolMessage(tool_call_id=call.id, content=run.output))
                continue
            match check_submit(run.submitted, last, locators_asked, rejected_before):
                case Accepted(args=args):
                    return solution(args, gate, step)
                case Rejected(message=reason, asked_locators=asked):
                    locators_asked = locators_asked or asked
                    rejected_before = rejected_before or not asked
                    output = f"{run.output}\n{reason}" if run.output else reason
                    messages.append(ToolMessage(tool_call_id=call.id, content=output))
    return FALLBACK.model_copy(update={"steps": MAX_STEPS})
