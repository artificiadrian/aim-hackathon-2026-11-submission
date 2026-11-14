"""The agent loop: one model works on the task's real files by writing Python, then calls a typed ``submit``.

The solver sees only the task and its files (path -> sha256 in the blob cache), never the Task API, so the same code
runs live (``mcnulty run``) and offline on saved runs (``mcnulty eval``). Actions come back as data, already checked;
the caller performs them. No task knowledge: the workspace's own operator files hold the rules. This module holds the
side effects (blob reads, model calls, sandbox runs); ``prompt`` and ``submit`` are pure.
"""

import base64

from pydantic import ValidationError

from mcnulty.agent.actions import verified_contacts
from mcnulty.agent.prompt import (
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
    ViewImageArgs,
    check_submit,
    solution,
    split_submission,
)
from mcnulty.blobs import BlobCache
from mcnulty.clients.arena import Task
from mcnulty.clients.openrouter import (
    AssistantMessage,
    Message,
    OpenRouterClient,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from mcnulty.frozen import Frozen
from mcnulty.record.events import TaskSolved
from mcnulty.record.log import Recorder

MAX_STEPS = 12  # model calls per task; the last one must submit
MAX_TOKENS = 3000  # per model reply
OPERATOR_FILES = ("/AGENTS.md", "/runbook.md")
IMAGE_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}


class TaskFiles(Frozen):
    """A task's files as the loop needs them, read once from the blob cache."""

    paths: dict[str, str]  # path -> sha256
    sizes: dict[str, int]
    trusted: dict[str, str]  # operator files and tickets: path -> text
    contacts: str  # /contacts.csv, empty when missing

    @classmethod
    def load(cls, files: dict[str, str], blobs: BlobCache):
        def text(path: str):
            data = blobs.get(files[path]) if path in files else None
            return data.decode("utf-8", errors="replace") if data is not None else ""

        trusted = [p for p in OPERATOR_FILES if p in files] + sorted(p for p in files if p.startswith("/tickets/"))
        return cls(
            paths=files,
            sizes={p: len(blobs.get(sha) or b"") for p, sha in files.items()},
            trusted={p: text(p) for p in trusted},
            contacts=text("/contacts.csv"),
        )

    @property
    def rules(self):
        return tuple(self.trusted[p] for p in OPERATOR_FILES if p in self.trusted)


class Tools(Frozen):
    """What the model's tools act on."""

    llm: OpenRouterClient
    box: TaskSandbox
    files: TaskFiles
    blobs: BlobCache

    def run_python(self, rec: Recorder, args: RunPythonArgs):
        result = self.box.run(rec, args.code)
        stdout, submitted = split_submission(result.stdout)
        output = script_output(stdout, result.stderr, result.exit_code, result.seconds)
        return ScriptRun(output=output, submitted=submitted)

    def view_image(self, rec: Recorder, args: ViewImageArgs):
        suffix = "." + args.path.rsplit(".", 1)[-1].lower()
        data = self.blobs.get(self.files.paths.get(args.path, ""))
        if data is None or suffix not in IMAGE_TYPES:
            return f"error: {args.path} is not an image file of the workspace"
        url = f"data:{IMAGE_TYPES[suffix]};base64,{base64.b64encode(data).decode()}"
        return self.llm.describe_image(rec, url, args.question, max_tokens=1500)

    def call(self, rec: Recorder, call: ToolCall):
        """Run one tool call. ``submit`` comes back as its arguments; a bad call becomes an error for the model."""
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
        return ScriptRun(output=f"error: no tool named {name}")


def solve(rec: Recorder, llm: OpenRouterClient, task: Task, files: dict[str, str], blobs: BlobCache):
    task_files = TaskFiles.load(files, blobs)
    box = TaskSandbox.create(rec, task.task_id, files, blobs)
    try:
        result = loop(rec, Tools(llm=llm, box=box, files=task_files, blobs=blobs), task)
    finally:
        box.remove()
    rec.log(TaskSolved(solution=result.model_dump(mode="json")))
    return result


def loop(rec: Recorder, tools: Tools, task: Task):
    files = tools.files
    verified = verified_contacts(files.contacts)
    messages: list[Message] = [
        SystemMessage(content=SYSTEM),
        UserMessage(content=first_message(task.instruction, files.trusted, files.sizes)),
    ]
    locators_asked = False
    for step in range(1, MAX_STEPS + 1):
        last = step == MAX_STEPS
        if last:
            messages.append(UserMessage(content=LAST_STEP))
        reply = tools.llm.chat(
            rec,
            shorten_old_outputs(tuple(messages)),
            tools=(SUBMIT,) if last else TOOLS,
            max_tokens=MAX_TOKENS,
            reasoning_effort="none",
            require_tool=last,
        )
        message = reply.choices[0].message
        messages.append(AssistantMessage(content=message.content, tool_calls=message.tool_calls))
        if not message.tool_calls:
            messages.append(UserMessage(content=USE_A_TOOL))
            continue
        for call in message.tool_calls:
            run = tools.call(rec, call)
            if run.submitted is None:
                messages.append(ToolMessage(tool_call_id=call.id, content=run.output))
                continue
            match check_submit(run.submitted, last, locators_asked):
                case Accepted(args=args):
                    return solution(args, verified, files.rules, step)
                case Rejected(message=reason, asked_locators=asked):
                    locators_asked = locators_asked or asked
                    output = f"{run.output}\n{reason}" if run.output else reason
                    messages.append(ToolMessage(tool_call_id=call.id, content=output))
    return FALLBACK.model_copy(update={"steps": MAX_STEPS})
