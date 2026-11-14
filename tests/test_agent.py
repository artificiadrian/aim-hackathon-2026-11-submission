"""The agent's pure parts (prompt, submit): submit parsing, the locator check, the file tree, the shortened history."""

import json

from mcnulty.agent.prompt import file_tree, shorten_old_outputs
from mcnulty.agent.submit import missing_locators, parse_submit, salvage_submit
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


def test_last_step_submit_keeps_outcome_and_answer_when_the_rest_is_invalid():
    args = salvage_submit('{"outcome": "OK", "answer": 5, "sources": [{"page": "x"}]}')
    assert args is not None
    assert (args.outcome, args.answer, args.sources) == ("OK", 5, ())
    assert salvage_submit('{"outcome": "MAYBE"}') is None
