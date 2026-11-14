from pathlib import Path

from pydantic import TypeAdapter

from mcnulty.record.events import Envelope, TaskCrashed
from mcnulty.record.log import EventLog, Recorder


class QuotaError(RuntimeError):
    status: int

    def __init__(self, status: int):
        super().__init__("quota reached")
        self.status = status


def fail_inner(limit: int):
    raise KeyError(limit)


def fail_outer():
    budget = 7
    try:
        fail_inner(budget)
    except KeyError as e:
        error = QuotaError(403)
        error.add_note("while reading the plan")
        raise error from e


def test_task_crashed_keeps_type_attributes_frames_locals_and_cause(tmp_path: Path):
    rec = Recorder(events=EventLog(run_dir=tmp_path, secrets=("arena_supersecretvalue",))).for_task("t07")
    try:
        fail_outer()
    except QuotaError as e:
        rec.log(TaskCrashed.from_exception(e))

    (line,) = (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    envelope = TypeAdapter(Envelope).validate_json(line)
    assert envelope.task_id == "t07"
    assert isinstance(envelope.event, TaskCrashed)
    error = envelope.event.error
    assert error.type.endswith("QuotaError")
    assert error.message == "quota reached"
    assert error.attributes == {"status": "403"}
    assert error.notes == ("while reading the plan",)
    assert [f.function for f in error.frames][-1] == "fail_outer"
    assert error.frames[-1].locals["budget"] == "7"
    assert error.cause is not None
    assert error.cause.type == "builtins.KeyError"
    assert error.cause.frames[-1].locals["limit"] == "7"
