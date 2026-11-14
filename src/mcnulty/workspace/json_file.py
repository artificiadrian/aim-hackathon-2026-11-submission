"""A JSON document: the parsed value plus its text lines (cited by path, or by line)."""

import json
from typing import Literal

from pydantic import JsonValue

from mcnulty.recorder import Frozen
from mcnulty.workspace.text_file import TextLine, text_lines


class JsonFile(Frozen):
    kind: Literal["json"] = "json"
    path: str
    lines: tuple[TextLine, ...]
    value: JsonValue


def parse_json(path: str, data: bytes):
    return JsonFile(path=path, lines=text_lines(path, data), value=json.loads(data))
