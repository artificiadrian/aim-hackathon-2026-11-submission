"""Plain text (md, txt and any UTF-8 file): cited by physical line."""

from typing import Literal

from mcnulty.arena import Source
from mcnulty.recorder import Frozen


class TextLine(Frozen):
    source: Source  # path + line (1-based, physical, blank lines count)
    text: str


class TextFile(Frozen):
    kind: Literal["text"] = "text"
    path: str
    lines: tuple[TextLine, ...]


def text_lines(path: str, data: bytes):
    """Physical lines, 1-based; blank lines count. A final newline ends the last line, it does not start a new one."""
    lines = data.decode("utf-8").split("\n")
    if lines and not lines[-1]:
        lines.pop()
    return tuple(TextLine(source=Source(path=path, line=n), text=line) for n, line in enumerate(lines, 1))


def parse_text(path: str, data: bytes):
    return TextFile(path=path, lines=text_lines(path, data))
