"""The loop's pure parts: submit parsing, the locator check, the file tree and the shortened history."""

from mcnulty.llm import Message, ToolMessage, UserMessage
from mcnulty.solver import file_tree, missing_locators, parse_submit, shorten_old_outputs


def test_submit_answer_given_as_json_text_becomes_its_value():
    args = parse_submit('{"outcome": "OK", "answer": "[\\"a\\", \\"b\\"]"}')
    assert args.answer == ["a", "b"]
    assert parse_submit('{"outcome": "OK", "answer": "2026-11-14T12:00:00Z"}').answer == "2026-11-14T12:00:00Z"


def test_sources_without_a_locator_are_found_except_path_only_types():
    args = parse_submit(
        '{"outcome": "OK", "answer": 1, "sources": [{"path": "/a.csv"}, {"path": "/b.csv", "record": "R-1"},'
        ' {"path": "/scan.png"}, {"path": "/registry.json"}]}'
    )
    assert missing_locators(args) == ["/a.csv"]


def test_file_tree_lists_small_folders_and_counts_large_ones():
    files = {f"/docs/a{i}.docx": "x" for i in range(13)} | {"/AGENTS.md": "y"}
    tree = file_tree(files, {"/AGENTS.md": 3874})
    assert tree.splitlines() == ["/AGENTS.md  (3,874 bytes)", "/docs/  (13 files: 13 .docx)"]


def test_old_tool_outputs_are_shortened_and_the_latest_kept():
    long = "x" * 2000
    messages: list[Message] = [UserMessage(content="task")]
    messages += [ToolMessage(tool_call_id=str(i), content=long) for i in range(3)]
    contents = [m.content for m in shorten_old_outputs(messages)]
    assert isinstance(contents[1], str)
    assert len(contents[1]) < 700
    assert contents[2:] == [long, long]
