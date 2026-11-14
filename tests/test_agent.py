"""The agent's pure parts (prompt, submit): submit parsing, the locator check, the file tree, the shortened history."""

import json

from mcnulty.agent.prompt import file_tree, shorten_old_outputs
from mcnulty.agent.submit import (
    Accepted,
    Rejected,
    check_submit,
    missing_locators,
    parse_submit,
    salvage_submit,
    split_submission,
)
from mcnulty.clients.openrouter import Message, ToolMessage, UserMessage


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


def test_a_script_submission_is_taken_from_the_output_and_checked():
    marker = '@@submit@@ {"outcome": "OK", "answer": [1], "sources": [{"path": "/a.csv", "record": "R-1"}]}'
    output, submitted = split_submission(f"rows: 3\n{marker}\ndone")
    assert output == "rows: 3\ndone"
    assert submitted is not None
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
