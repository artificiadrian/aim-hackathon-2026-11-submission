"""JSON Lines: one value per physical line, cited by line."""

import json
from typing import Literal

from pydantic import JsonValue

from mcnulty.clients.arena import Source
from mcnulty.frozen import Frozen
from mcnulty.workspace.text_file import text_lines


class JsonLine(Frozen):
    """``value`` is None when the line is not valid JSON; ``text`` is always the line as written."""

    source: Source  # path + line
    text: str
    value: JsonValue


class JsonLinesFile(Frozen):
    kind: Literal["jsonl"] = "jsonl"
    path: str
    lines: tuple[JsonLine, ...]


def parse_line(text: str):
    try:
        return json.loads(text) if text.strip() else None
    except ValueError:
        return None


def parse_jsonl(path: str, data: bytes):
    lines = tuple(JsonLine(source=t.source, text=t.text, value=parse_line(t.text)) for t in text_lines(path, data))
    return JsonLinesFile(path=path, lines=lines)
