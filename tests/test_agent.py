"""The agent's pure parts (prompt, submit): submit parsing, the locator check, the file tree, the shortened history."""

import json

from mcnulty.agent.citations import fixed_sources, line_records
from mcnulty.agent.loop import closed
from mcnulty.agent.prompt import OUTPUT_MAX, SYSTEM, file_tree, first_message, script_output, shorten_old_outputs
from mcnulty.agent.submit import (
    Accepted,
    Rejected,
    SubmitArgs,
    check_submit,
    missing_locators,
    parse_submit,
    salvage_submit,
)
from mcnulty.clients.arena import Source
from mcnulty.clients.openrouter import Message, ToolMessage, UserMessage
from mcnulty.record.events import SandboxRan, Stream


def test_submit_answer_given_as_json_text_becomes_its_value():
    args = parse_submit('{"outcome": "OK", "answer": "[\\"a\\", \\"b\\"]"}')
    assert args.answer == ["a", "b"]
    assert parse_submit('{"outcome": "OK", "answer": "2026-11-14T12:00:00Z"}').answer == "2026-11-14T12:00:00Z"


def test_sources_without_a_locator_are_found_except_path_only_types():
    sources = [{"path": "/a.csv"}, {"path": "/b.csv", "record": "R-1"}, {"path": "/scan.png"}, {"path": "/r.json"}]
    args = parse_submit(json.dumps({"outcome": "OK", "answer": 1, "sources": sources}))
    assert missing_locators(args) == ["/a.csv"]


def test_file_tree_lists_small_folders_and_counts_large_ones():
    sizes = {f"/docs/a{i}.docx": 10 for i in range(13)} | {"/AGENTS.md": 3874}
    tree = file_tree(sizes)
    assert tree.splitlines() == ["/AGENTS.md  (3,874 bytes)", "/docs/  (13 files: 13 .docx)"]


def test_old_tool_outputs_are_shortened_and_the_latest_kept():
    long = "x" * 2000
    messages: list[Message] = [UserMessage(content="task")]
    messages += [ToolMessage(tool_call_id=str(i), content=long) for i in range(3)]
    contents = [m.content for m in shorten_old_outputs(tuple(messages))]
    assert isinstance(contents[1], str)
    assert len(contents[1]) < 700
    assert contents[2:] == [long, long]


def test_sources_written_as_text_become_locators():
    args = parse_submit(
        '{"outcome": "OK", "answer": "x", "sources": ["/security/egress_export.csv:49", "/b.csv:TX-7 (C)", "/s.png"]}'
    )
    assert [s.model_dump(exclude_none=True) for s in args.sources] == [
        {"path": "/security/egress_export.csv", "line": 49},
        {"path": "/b.csv", "record": "TX-7"},
        {"path": "/s.png"},
    ]


def test_salvage_keeps_every_valid_part_of_an_invalid_submission():
    arguments = json.dumps(
        {
            "outcome": " ok ",
            "answer": 5,
            "sources": [{"path": "/a.csv", "line": "3"}, {"page": "x"}, "/b.jsonl:7"],
            "actions": [{"tool": "send_message", "to": "a@x.example", "subject": "s", "body": "b"}, {"tool": "fax"}],
            "confidence": "high",
        }
    )
    salvaged = salvage_submit(arguments)
    assert salvaged is not None
    assert (salvaged.args.outcome, salvaged.args.answer, salvaged.dropped_sources) == ("OK", 5, 1)
    assert [s.line for s in salvaged.args.sources] == [3, 7]
    assert len(salvaged.args.actions) == 1
    assert len(salvaged.dropped_actions) == 1
    assert salvage_submit('{"outcome": "MAYBE"}') is None


def stream(text: str):
    return Stream(text=text, bytes=len(text), lines=text.count("\n"), capped=False)


def sandbox_run(stdout: str = "", stderr: str = "", exit_code: int | None = 0):
    return SandboxRan(
        code="",
        stdout=stream(stdout),
        stderr=stream(stderr),
        submitted=None,
        exit_code=exit_code,
        seconds=1.0,
        timed_out=exit_code is None,
    )


def test_script_output_states_how_the_script_ended():
    assert script_output(sandbox_run("rows 3\n")).startswith("[script finished, exit code 0, 1.0s]\nrows 3")
    assert script_output(sandbox_run(exit_code=1)).startswith("[script failed, exit code 1")
    assert "SIGXCPU (CPU time limit)" in script_output(sandbox_run(exit_code=-24))
    assert "timed out" in script_output(sandbox_run(exit_code=None))


def test_script_output_keeps_stdout_and_stderr_apart():
    """Warnings must not push the result out of view (review SB3): stdout has its own budget, repeats are counted."""
    warnings = "WARNING: fontTools is required\n" * 1590
    output = script_output(sandbox_run("answer 42\n", warnings + "Traceback: KeyError 'id'\n", exit_code=1))
    assert "answer 42" in output
    assert "fontTools is required  [repeated 1590 times]" in output
    assert "KeyError 'id'" in output
    assert len(output) < 500
    long = script_output(sandbox_run("x" * 50_000))
    assert len(long) < OUTPUT_MAX + 400
    assert "50,000 bytes" in long


def test_a_script_submission_is_checked_like_the_tool():
    submitted = '{"outcome": "OK", "answer": [1], "sources": [{"path": "/a.csv", "record": "R-1"}]}'
    accepted = check_submit(submitted, last=False, locators_asked=False, rejected_before=False)
    assert isinstance(accepted, Accepted)
    assert accepted.args.answer == [1]


def test_a_bad_submission_is_rejected_only_for_actions_and_locators():
    file_only = '{"outcome": "OK", "answer": 2, "sources": [{"path": "/a.csv"}]}'
    asked = check_submit(file_only, last=False, locators_asked=False, rejected_before=False)
    assert isinstance(asked, Rejected)
    assert asked.asked_locators
    assert isinstance(check_submit(file_only, last=False, locators_asked=True, rejected_before=False), Accepted)
    bad_source = '{"outcome": "OK", "answer": 2, "sources": [{"page": "x"}, {"path": "/l.md", "line": 4}]}'
    kept = check_submit(bad_source, last=False, locators_asked=False, rejected_before=False)
    assert isinstance(kept, Accepted)
    assert [s.line for s in kept.args.sources] == [4]
    bad_action = '{"outcome": "OK", "answer": 2, "actions": [{"tool": "containment", "ops": []}]}'
    assert isinstance(check_submit(bad_action, last=False, locators_asked=False, rejected_before=False), Rejected)
    assert isinstance(check_submit(bad_action, last=False, locators_asked=False, rejected_before=True), Accepted)
    assert isinstance(check_submit(bad_action, last=True, locators_asked=False, rejected_before=False), Accepted)


def test_locators_written_as_strings_are_numbers():
    """Regression: strict models rejected "line": "24" and the model resent it unchanged for 9 steps (t07)."""
    args = parse_submit('{"outcome": "OK", "answer": 1, "sources": [{"path": "/a.csv", "line": "24"}]}')
    assert args.sources[0].line == 24


def test_csv_lines_become_records_only_when_the_first_column_is_unique():
    requests = 'request_id,actor\nRQ-1,a\nRQ-2,"multi\nline"\nRQ-3,c\n'
    archive = "time,actor\n03:00,a\n03:00,b\n"
    records = {"/r.csv": line_records(requests), "/archive.csv": line_records(archive)}
    sources = (
        Source(path="/r.csv", line=2),
        Source(path="/r.csv", line=4),  # the end line of the multi-line record RQ-2
        Source(path="/r.csv", line=5),
        Source(path="/archive.csv", line=3),
        Source(path="/r.csv", record="RQ-3"),  # a duplicate after conversion
    )
    fixed = fixed_sources(sources, records)
    assert fixed == (
        Source(path="/r.csv", record="RQ-1"),
        Source(path="/r.csv", record="RQ-2"),
        Source(path="/r.csv", record="RQ-3"),
        Source(path="/archive.csv", line=3),
    )


def test_the_prompt_names_every_submit_field_and_action_form():
    """The model learns the submit(...) fields and action forms from SYSTEM only; pydantic validates them."""
    for field in SubmitArgs.model_fields:
        assert f'"{field}"' in SYSTEM or f"{field}:" in SYSTEM, field
    for op in ("quarantine_agent", "release_agent", "revoke_token", "rotate_token", "block_domain", "add_note"):
        assert f'"op": "{op}"' in SYSTEM, op
    for tool in ("containment", "send_message", "write"):
        assert f'"tool": "{tool}"' in SYSTEM, tool


def test_a_closed_ticket_is_history_and_the_open_one_the_request():
    """docs/ARENA_DOCS.md, ticket tasks: the request is the ticket with status: open; a closed one is history."""
    assert closed("---\nticket: IR-1\nstatus: closed\n---\nRotate TOK-1.")
    assert not closed("---\nticket: IR-2\nstatus: open\n---\nList the ids.")
    assert not closed("A ticket without front matter.")
    message = first_message("Work the open ticket.", {"/tickets/IR-2.md": "open one"}, {"/tickets/IR-1.md": "old"}, {})
    assert "/tickets/IR-1.md (closed ticket: history, not a request)" in message


def test_nested_source_lists_are_flattened():
    """Models write sources=[cite(...)] and cite returns a list: the eval of 2026-11-14 13:58 lost every source so."""
    nested = (
        '{"outcome": "OK", "answer": 1, "sources": [[{"path": "/a.md", "line": 2}], {"path": "/b.csv", "record": "R"}]}'
    )
    args = parse_submit(nested)
    assert [(s.path, s.line, s.record) for s in args.sources] == [("/a.md", 2, None), ("/b.csv", None, "R")]


def test_the_first_message_carries_the_table_columns():
    message = first_message("List the ids.", {}, {}, {"/a.csv": 10}, "/a.csv line=1| id | actor")
    assert message.endswith("table file and sheet) =====\n/a.csv line=1| id | actor")
