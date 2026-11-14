"""The sandbox helpers on the host (the dev venv has packages/sandbox): every citation they make is one the harness
accepts, and the helpers read every format. The chroot side is in tests/test_sandbox.py."""

import json
from pathlib import Path

import pytest
import ws.formats
from ws.files import cite, columns, grep
from ws.models import Source as WsSource

from mcnulty.clients.arena import Source


@pytest.fixture
def files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(ws.formats, "CACHE", tmp_path / ".ws")  # the parse cache is the sandbox's /tmp, not the host's
    (tmp_path / "r.csv").write_text("request_id,actor\nRQ-1,ext\nRQ-2,int\n", encoding="utf-8")
    (tmp_path / "t.csv").write_text("time,actor\n03:00,ext\n03:00,int\n", encoding="utf-8")
    (tmp_path / "a.jsonl").write_text('{"a": 1}\n{"agent": "devops-07"}\n', encoding="utf-8")
    (tmp_path / "x.json").write_text(json.dumps([{"id": "X-1"}, {"id": "X-2", "agent": "devops-07"}]), encoding="utf-8")
    (tmp_path / "n.md").write_text("# Notes\n\ndevops-07 was seen\n", encoding="utf-8")
    return tmp_path


def test_every_citation_of_the_helpers_is_one_the_harness_accepts(files: Path):
    found = [*cite(str(files / "r.csv"), "RQ-2", "request_id"), *cite(str(files / "t.csv"), "int")]
    found += [hit.source for hit in grep("devops-07", str(files))]
    assert {s.model_dump_json(exclude_none=True) for s in found} == {
        WsSource(path=str(files / "r.csv"), record="RQ-2").model_dump_json(exclude_none=True),
        WsSource(path=str(files / "t.csv"), line=3).model_dump_json(exclude_none=True),
        WsSource(path=str(files / "a.jsonl"), line=2).model_dump_json(exclude_none=True),
        WsSource(path=str(files / "x.json"), record="X-2").model_dump_json(exclude_none=True),
        WsSource(path=str(files / "n.md"), line=3).model_dump_json(exclude_none=True),
    }
    for source in found:
        accepted = Source.model_validate_json(source.model_dump_json(exclude_none=True))
        assert accepted.model_dump(exclude_none=True) == source.model_dump(exclude_none=True)


def test_cite_with_a_column_a_file_does_not_have_still_cites(files: Path, capsys: pytest.CaptureFixture[str]):
    """A missing column must not raise: the eval of 2026-11-14 13:58 lost t09's whole submit(...) to it."""
    found = cite(str(files / "a.jsonl"), "devops-07", column="ts")
    assert [s.line for s in found] == [2]
    assert "no column 'ts'" in capsys.readouterr().err
    assert [s.record for s in cite(str(files / "r.csv"), "int", column="actor")] == ["RQ-2"]


def test_columns_lists_the_header_of_every_table(files: Path):
    lines = columns(str(files)).splitlines()
    assert f"{files / 'r.csv'} line=1| request_id | actor" in lines
    assert f"{files / 'a.jsonl'} line=1| keys: a" in lines
    assert all("n.md" not in line for line in lines)
