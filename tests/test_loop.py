"""The agent loop with a scripted model (httpx mock transport) and a fake sandbox: tool dispatch, submit from a
script, bad arguments, cut replies, the locator re-ask, the forced last step and the fallback."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import override

import httpx
import pytest

from mcnulty.agent.loop import MAX_STEPS, NO_DEADLINE, TaskFiles, Tools, loop
from mcnulty.agent.sandbox import SandboxPool, TaskSandbox
from mcnulty.blobs import BlobCache
from mcnulty.clients.arena import Task
from mcnulty.clients.openrouter import ChatRequest, ChatResponse, CostTracker, OpenRouterClient
from mcnulty.record.events import SandboxRan, Stream
from mcnulty.record.log import EventLog, Recorder


def stream(text: str):
    return Stream(text=text, bytes=len(text), lines=text.count("\n"), capped=False)


@dataclass(frozen=True, slots=True)
class FakeBox(TaskSandbox):
    """Runs nothing: each script's output and submit(...) JSON come from ``results`` in order."""

    results: list[tuple[str, str | None]] = field(default_factory=list[tuple[str, str | None]])
    codes: list[str] = field(default_factory=list[str])

    @override
    def run(self, rec: Recorder, code: str, timeout: float = 30.0, wait: float = 90.0):
        self.codes.append(code)
        out, submitted = self.results.pop(0)
        return SandboxRan(
            code=code,
            stdout=stream(out),
            stderr=stream(""),
            submitted=submitted,
            exit_code=0,
            seconds=0.1,
            timed_out=False,
        )


def call(name: str, arguments: str, finish: str = "tool_calls"):
    tool_call = {"id": f"c-{name}", "type": "function", "function": {"name": name, "arguments": arguments}}
    message = {"role": "assistant", "content": None, "tool_calls": [tool_call]}
    return ChatResponse.model_validate({"choices": [{"message": message, "finish_reason": finish}]})


def say(text: str):
    message = {"role": "assistant", "content": text}
    return ChatResponse.model_validate({"choices": [{"message": message, "finish_reason": "stop"}]})


def python(code: str, finish: str = "tool_calls"):
    return call("run_python", json.dumps({"code": code}), finish)


SUBMIT_OK = json.dumps({"outcome": "OK", "answer": [1], "sources": [{"path": "/a.csv", "record": "R-1"}]})


@dataclass
class Scripted:
    """The model: replies in order; records each request body."""

    replies: list[ChatResponse]
    requests: list[ChatRequest] = field(default_factory=list[ChatRequest])

    def __call__(self, request: httpx.Request):
        self.requests.append(ChatRequest.model_validate_json(request.content))
        return httpx.Response(200, content=self.replies.pop(0).model_dump_json())


def tool_names(request: ChatRequest):
    return [t.function.name for t in request.tools or ()]


def run_loop(tmp_path: Path, replies: list[ChatResponse], results: list[tuple[str, str | None]]):
    model = Scripted(replies)
    llm = OpenRouterClient(
        http=httpx.Client(transport=httpx.MockTransport(model)), base_url="http://or.test/v1", tracker=CostTracker()
    )
    box = FakeBox(root=tmp_path, uid=10001, pool=SandboxPool(), results=results)
    files = TaskFiles(
        paths={"/a.csv": "x"}, sizes={"/a.csv": 10}, trusted={"/AGENTS.md": "rules"}, contacts="", records={}
    )
    tools = Tools(llm=llm, box=box, files=files, blobs=BlobCache(root=tmp_path / "blobs"))
    rec = Recorder(events=EventLog(run_dir=tmp_path / "run"))
    solution = loop(rec, tools, Task(task_id="t01", instruction="List the ids."), NO_DEADLINE)
    return solution, model, box


def test_a_script_that_submits_ends_the_task(tmp_path: Path):
    solution, model, box = run_loop(
        tmp_path, [python("print(rows)"), python("submit(...)")], [("3 rows", None), ("", SUBMIT_OK)]
    )
    assert (solution.submission.outcome, solution.submission.answer, solution.steps) == ("OK", [1], 2)
    assert box.codes == ["print(rows)", "submit(...)"]
    assert tool_names(model.requests[0]) == ["run_python", "view_image"]


def test_bad_arguments_and_text_replies_go_back_to_the_model(tmp_path: Path):
    replies = [call("run_python", "{not json"), say("I think the answer is 1"), python("submit(...)")]
    solution, model, _ = run_loop(tmp_path, replies, [("", SUBMIT_OK)])
    assert solution.steps == 3
    second = model.requests[1].model_dump_json()
    assert "error in the arguments of run_python" in second
    assert "Use a tool" in model.requests[2].model_dump_json()


def test_a_cut_reply_runs_nothing(tmp_path: Path):
    replies = [python("print(1", finish="length"), python("submit(...)")]
    solution, model, box = run_loop(tmp_path, replies, [("", SUBMIT_OK)])
    assert box.codes == ["submit(...)"]
    assert solution.steps == 2
    assert "cut at the token limit" in model.requests[1].model_dump_json()


def test_a_source_without_locator_is_asked_for_once(tmp_path: Path):
    file_only = json.dumps({"outcome": "OK", "answer": [1], "sources": [{"path": "/a.csv"}]})
    replies = [python("submit(a)"), python("submit(b)")]
    solution, model, _ = run_loop(tmp_path, replies, [("", file_only), ("", file_only)])
    assert solution.steps == 2  # the second file-only submit is taken
    assert "no locator" in model.requests[1].model_dump_json()


@pytest.mark.parametrize("submits_last", [True, False])
def test_the_last_step_must_run_a_script_else_the_fallback(tmp_path: Path, submits_last: bool):
    replies = [say("thinking")] * (MAX_STEPS - 1) + [python("submit(...)")]
    results: list[tuple[str, str | None]] = [("", SUBMIT_OK if submits_last else None)]
    solution, model, _ = run_loop(tmp_path, replies, results)
    last = model.requests[-1]
    assert last.tool_choice == "required"
    assert tool_names(last) == ["run_python"]
    assert solution.steps == MAX_STEPS
    assert solution.submission.outcome == ("OK" if submits_last else "UNSUPPORTED")
