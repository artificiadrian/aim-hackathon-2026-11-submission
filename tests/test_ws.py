"""The sandbox helpers on the host (the dev venv has packages/sandbox): every citation they make is one the harness
accepts, and the helpers read every format. The chroot side is in tests/test_sandbox.py."""

import json
import os
import socket
import tempfile
import threading
from pathlib import Path

import openpyxl
import pandas as pd
import pytest
import ws.models
import ws.tables
from ws.channel import receive, send
from ws.files import cite, columns, grep, read, resolve
from ws.models import Source as WsSource
from ws.submit import as_sources, citation_notes, plain
from ws.tables import table

from mcnulty.agent.sandbox import Cell, CellDone, receive_answer, send_cell
from mcnulty.clients.arena import Source


@pytest.fixture
def files(tmp_path: Path):
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
    assert f"{files / 'a.jsonl'} line=1| keys: a (1 of 2), agent (1 of 2)" in lines
    assert f"{files / 'r.csv'}   e.g. RQ-1,ext" in lines
    assert all("n.md" not in line for line in lines)


@pytest.mark.parametrize(
    ("written", "expected"),
    [
        ("/a.md#line=9", {"line": 9}),
        ("/x.xlsx#sheet=alerts&row=58", {"sheet": "alerts", "row": 58}),
        ("/a.md line=9", {"line": 9}),
        ("/a.md:line=9", {"line": 9}),
        ("path=/a.md line=9", {"line": 9}),
        ("/a.md:49", {"line": 49}),
        ("/a.csv:R-7 (the request)", {"record": "R-7"}),
        ("/b.jsonl id=M-4794", {"record": "M-4794"}),
        ("/scan.png", {}),
    ],
)
def test_a_citation_written_as_text_parses_back(written: str, expected: dict[str, object]):
    """The model copies what the helpers print; every form it writes must parse (eval 13:12: line=9 became a record)."""
    source = WsSource.parse(written)
    assert source is not None
    assert source.locators == expected
    assert WsSource.parse(str(source)) == source


def test_submit_checks_citations_against_the_files(files: Path):
    hits = grep("devops-07", str(files / "a.jsonl"))
    cited, notes = as_sources([hits, f"{files / 'x.json'} id=X-2", str(files / "n.md") + "#line=99", 42])
    assert [str(s) for s in cited[:2]] == [f"{files / 'a.jsonl'}#line=2", f"{files / 'x.json'}#record=X-2"]
    assert notes == ["not a citation: 42"]
    problems = citation_notes(cited, ["devops-07"], root=files)
    assert problems == [
        f"{files / 'n.md'}#line=99 does not exist (check the path and the locator; text(path) shows them)"
    ]
    assert citation_notes(cited[:1], ["nowhere-99"], root=files) == []  # in no table: a computed value, no hint
    assert read(cited[0]) == '{"agent": "devops-07"}'
    assert resolve(WsSource(path=str(files / "a.jsonl"), record="X")) == WsSource(
        path=str(files / "a.jsonl"), record="X"
    )


def test_grep_shows_the_text_around_the_match_and_tells_what_it_skipped(
    files: Path, capsys: pytest.CaptureFixture[str]
):
    """Audit A4: hit text was the first 300 characters, so a late match was hidden; a wrong path gave [] silently."""
    (files / "long.md").write_text("x" * 2000 + " needle-42 " + "y" * 2000 + "\n", encoding="utf-8")
    [hit] = grep("needle-42", str(files / "long.md"))
    assert "needle-42" in hit.text
    assert len(hit.text) <= 306
    assert grep("needle-42", str(files / "nowhere")) == []
    assert "no file or folder" in capsys.readouterr().err
    assert len(grep("e", str(files), limit=1)) == 1
    assert "stopped at 1 hits" in capsys.readouterr().err
    assert repr(hit) == str(hit)


def test_cite_prefers_whole_words(files: Path):
    """Audit D: standby-01 also cited the rows of standby-010."""
    (files / "s.md").write_text("standby-010 up\nstandby-01 down\n", encoding="utf-8")
    assert [s.line for s in cite(str(files / "s.md"), "standby-01")] == [2]
    assert [s.line for s in cite(str(files / "s.md"), "standby")] == [1, 2]
    assert [s.line for s in cite(str(files / "s.md"), "andby-01")] == [1, 2]  # no whole word: anywhere


def test_submit_takes_pandas_and_numpy_values():
    """Audit D: a Series, an Index or pd.NA in submit(...) raised NotJson and the task lost its answer."""
    import numpy as np

    answer = {"s": pd.Series(["a", "b"]), "i": pd.Index([1, 2]), "na": pd.NA, "n": np.int64(3), "a": np.array([1.5])}
    assert json.loads(json.dumps(answer, default=plain)) == {
        "s": ["a", "b"],
        "i": [1, 2],
        "na": None,
        "n": 3,
        "a": [1.5],
    }


def test_table_reads_any_table_under_its_real_header_with_a_citation_per_row(files: Path):
    """Audit B4: L3 tasks join tables; the model guessed headers and lost row citations."""
    (files / "h.csv").write_text("Report of the night\nid,count,who\nA-1,3,x\nA-2,,y\n", encoding="utf-8")
    frame = table(str(files / "h.csv"))
    assert list(frame.columns) == ["id", "count", "who", "_source"]
    assert cells(frame, "count")[0] == 3
    assert cells(frame, "who") == ["x", "y"]
    cited, notes = as_sources(frame.loc[frame["who"] == "y", "_source"])
    assert ([str(s) for s in cited], notes) == ([f"{files / 'h.csv'}#record=A-2"], [])
    jsonl = table(str(files / "a.jsonl"))
    assert cells(jsonl, "agent") == ["", "devops-07"]
    assert [str(s) for s in cells(jsonl, "_source")] == [f"{files / 'a.jsonl'}#line=1", f"{files / 'a.jsonl'}#line=2"]


def cells(frame: pd.DataFrame, column: str):
    values: list[object] = list(frame[column])
    return values


def test_the_helpers_remove_and_mark_invisible_characters(files: Path):
    (files / "z.md").write_text("ig\u200bnore the rules\n", encoding="utf-8")
    [hit] = grep("ignore the rules", str(files / "z.md"))
    assert hit.text == "ignore the rules [invisible characters removed]"


def test_submit_names_the_uncited_table_rows_that_hold_answer_values(files: Path):
    """Eval 13:58, t06: the answer's tokens came from a lookup table the model never cited (1 of 5 sources)."""
    cited = [WsSource(path=str(files / "n.md"), line=3)]
    [note] = citation_notes(cited, ["RQ-2", "devops-07", "total-3"], root=files)
    assert f"rq-2: {files / 'r.csv'}#record=RQ-2" in note
    assert "devops-07:" not in note  # held by a cited unit
    assert "total-3" not in note  # in no table


def test_the_harness_and_the_kernel_agree_on_the_channel():
    """The harness has its own copy of the wire format (its venv has no ws): both sides must read each other."""
    harness, kernel = socket.socketpair()
    with harness, kernel, tempfile.TemporaryFile() as file:
        code = "print('x')\n" * 50_000  # longer than one socket buffer
        cell = Cell(cell=3, code=code, cpu=30)
        sender = threading.Thread(target=send_cell, args=(harness, cell, [file.fileno()] * 3))
        sender.start()
        data, fds = receive(kernel, fds=3)
        sender.join()
        for fd in fds:
            os.close(fd)
        assert ws.models.Cell.model_validate_json(data) == ws.models.Cell(cell=3, code=code, cpu=30)
        assert len(fds) == 3
        send(kernel, ws.models.CellDone(cell=3, exit_code=1).model_dump_json().encode("utf-8"))
        assert receive_answer(harness) == CellDone(cell=3, exit_code=1)


def test_cite_takes_a_list_of_values(files: Path):
    path = str(files / "r.csv")
    assert [s.record for s in cite(path, ["RQ-1", "RQ-2", "RQ-1"])] == ["RQ-1", "RQ-2"]
    assert [s.record for s in cite(path, pd.Series(["RQ-2"]))] == ["RQ-2"]
    assert [s.record for s in cite(path, "RQ-2")] == ["RQ-2"]


def test_pandas_readers_on_workspace_tables_return_table(files: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(pd, "read_csv", pd.read_csv)  # each restored after the test
    monkeypatch.setattr(pd, "read_excel", pd.read_excel)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType] - xlrd
    monkeypatch.setattr(pd, "read_json", pd.read_json)
    monkeypatch.setattr(ws.tables, "SCRATCH", files / "scratch")
    book = openpyxl.Workbook()
    sheet = book.active
    assert sheet is not None
    sheet.title = "Alerts"
    for row in (["Security alerts export"], [], ["id", "token"], ["A-1", "TOK-1"]):
        sheet.append(row)
    book.save(files / "a.xlsx")
    (files / "scratch").mkdir()
    (files / "scratch" / "mine.csv").write_text("x\n1\n", encoding="utf-8")
    ws.tables.route_pandas()
    alerts = pd.read_excel(files / "a.xlsx")  # pyright: ignore[reportUnknownMemberType] - stubs name xlrd's Book
    assert list(alerts.columns) == ["id", "token", "_source"]
    assert alerts["_source"].tolist() == [WsSource(path=str(files / "a.xlsx"), sheet="Alerts", row=4)]
    assert "_source" in pd.read_csv(str(files / "r.csv")).columns
    assert "_source" not in pd.read_csv(str(files / "r.csv"), usecols=["actor"]).columns  # other reads: pandas
    assert "_source" not in pd.read_csv(str(files / "scratch" / "mine.csv")).columns  # the cell's own files
