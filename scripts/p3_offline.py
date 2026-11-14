"""P3 offline check ($0): every submission the citation check sent back, replayed through ``ws.submit.check_citations``
in the task's sandbox and graded against the expected sources. Runs only in the image (AGENTS.md, Commands):

docker run --rm -v "$PWD/src:/app/src" -v "$PWD/runs:/app/runs:ro" -v "$PWD/.cache:/app/.cache" \\
  -v "$PWD/scripts:/app/scripts:ro" "${WS[@]}" --entrypoint uv mcnulty run --frozen \\
  python scripts/p3_offline.py runs/<saved run> runs/<eval run> ...
"""

import sys
import tempfile
from pathlib import Path

from pydantic import BaseModel, JsonValue, TypeAdapter

from mcnulty.agent.sandbox import SandboxPool, TaskSandbox
from mcnulty.blobs import BlobCache
from mcnulty.clients.arena import Source
from mcnulty.commands.evaluate import SavedRun
from mcnulty.commands.grading import source_hit
from mcnulty.record.events import Envelope, SandboxRan
from mcnulty.record.log import EventLog, Recorder

SOURCES = TypeAdapter(list[Source])
BLOBS = Path("/app/.cache/blobs")


class Payload(BaseModel):
    """One ``submit(...)`` line from the sandbox (``ws.submit``)."""

    answer: JsonValue = None
    sources: list[Source] = []
    notes: list[str] = []


class Checked(BaseModel):
    """``ws.submit.Checked`` as JSON."""

    sources: list[Source]
    added: list[Source] = []
    notes: list[str] = []


def hits(got: list[Source], want: list[Source]):
    return sum(any(source_hit(g, w) for g in got) for w in want)


def replay(rec: Recorder, pool: SandboxPool, saved: SavedRun, task: str, payload: Payload):
    box = TaskSandbox.create(rec, pool, f"p3-{task}", saved.files(task), BlobCache(root=BLOBS))
    try:
        code = (
            "from ws.formats import JSON\nfrom ws.models import Source\nfrom ws.submit import check_citations\n"
            f"sources = [Source.model_validate_json(s) for s in {[s.model_dump_json() for s in payload.sources]!r}]\n"
            f"answer = JSON.validate_json({TypeAdapter(JsonValue).dump_json(payload.answer).decode()!r})\n"
            "print(check_citations(sources, answer).model_dump_json())"
        )
        ran = box.run(rec, code, timeout=60)
    finally:
        box.remove()
    if ran.exit_code != 0:
        print(task, "FAILED", ran.stderr.text[-300:])
        return None
    return Checked.model_validate_json(ran.stdout.text)


def main():
    saved = SavedRun.load(Path(sys.argv[1]))
    rec = Recorder(events=EventLog(run_dir=Path(tempfile.mkdtemp())))
    pool = SandboxPool()
    rows: list[tuple[str, str, int, int, int, int, int, int]] = []
    for run in sys.argv[2:]:
        for line in Path(run, "events.jsonl").read_text(encoding="utf-8").splitlines():
            logged = Envelope.model_validate_json(line)
            event, task = logged.event, logged.task_id
            if not isinstance(event, SandboxRan) or event.submitted is None or task is None:
                continue
            payload = Payload.model_validate_json(event.submitted)
            if not payload.notes:
                continue
            checked = replay(rec, pool, saved, task, payload)
            if checked is None:
                continue
            want = SOURCES.validate_python(saved.expected(task).expected_sources or [])
            right = sum(any(source_hit(a, w) for w in want) for a in checked.added)
            before, after = hits(payload.sources, want), hits(checked.sources, want)
            rows.append((Path(run).name, task, len(want), before, after, len(checked.added), right, len(checked.notes)))
    print("run task expected hits_before hits_after added added_right notes_left")
    for row in rows:
        print(*row)
    print(
        f"hits {sum(r[3] for r in rows)} -> {sum(r[4] for r in rows)}, added {sum(r[5] for r in rows)} "
        f"({sum(r[6] for r in rows)} right), still sent back {sum(1 for r in rows if r[7])} of {len(rows)}"
    )


if __name__ == "__main__":
    main()
