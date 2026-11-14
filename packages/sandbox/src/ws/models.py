"""What the helpers return: pydantic models, so they print, serialize and go into submit(...) as they are."""

import re
from typing import ClassVar, override

from pydantic import BaseModel, ConfigDict

PATH = re.compile(r"(?:path\s*[=:]\s*)?(/[^\s#:|,;&'\"]+)")
LOCATOR = re.compile(r"\b(line|record|id|sheet|row|paragraph|page)\s*[=:]\s*([^\s&,|;#'\"]+)")
NUMBERED = frozenset({"line", "row", "paragraph", "page"})


class Source(BaseModel):
    """A citation: the path plus the locator the platform grades for that file type (docs/ARENA_DOCS.md).

    One flat model with every locator optional, the platform's own shape. A union (one class per locator kind) would
    reject impossible mixes; revisit if runs show citations with wrong or mixed locators.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    path: str
    line: int | None = None
    record: str | None = None
    sheet: str | None = None
    row: int | None = None
    paragraph: int | None = None
    page: int | None = None

    @property
    def locators(self):
        fields = {
            "line": self.line,
            "record": self.record,
            "sheet": self.sheet,
            "row": self.row,
            "paragraph": self.paragraph,
            "page": self.page,
        }
        return {k: v for k, v in fields.items() if v is not None}

    @property
    def where(self):
        """The locator as text, e.g. ``sheet=Alerts row=58``; ``file`` for a path-only citation."""
        return " ".join(f"{k}={v}" for k, v in self.locators.items()) or "file"

    @override
    def __str__(self):
        """The exact printed form, ``/a.xlsx#sheet=alerts&row=58``; ``Source.parse`` reads it back."""
        query = "&".join(f"{k}={v}" for k, v in self.locators.items())
        return f"{self.path}#{query}" if query else self.path

    @classmethod
    def parse(cls, text: str):
        """A citation written as text: the exact form ``/a.csv#record=R-7``, and the loose forms models write:
        ``/a.md line=9``, ``/a.md:line=9``, ``path=/a.md line=9``, ``/a.md:49`` (a line), ``/a.csv:R-7`` (a
        record), ``/b.jsonl id=M-1`` (a record). None when there is no workspace path in it."""
        found = PATH.search(text)
        if found is None:
            return None
        path, rest = found.group(1), text[found.end() :]
        pairs = {("record" if m.group(1) == "id" else m.group(1)): m.group(2) for m in LOCATOR.finditer(rest)}
        if not pairs:
            bare = rest.lstrip(":").split(" (")[0].strip().strip(",;|")
            if bare.isdigit():
                pairs = {"line": bare}
            elif bare:
                pairs = {"record": bare}
        numbers = {k: int(v) for k, v in pairs.items() if k in NUMBERED and v.isdigit()}
        texts = {k: v for k, v in pairs.items() if k not in NUMBERED}
        return cls.model_validate({"path": path, **texts, **numbers})


class Unit(BaseModel):
    """One citable unit of a file: a line, a csv record, a sheet row, a paragraph, a page."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    source: Source
    text: str
    cells: tuple[str, ...] = ()  # the cells of a table row, in column order


class Shell(BaseModel):
    """What ``sh`` returns: both streams apart and the exit code (None: timed out)."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    stdout: str
    stderr: str
    exit_code: int | None

    @override
    def __str__(self):
        status = "" if self.exit_code == 0 else f"\n[exit code {self.exit_code}]"
        return self.stdout + (f"\n[stderr]\n{self.stderr}" if self.stderr else "") + status


class Hit(BaseModel):
    """A search hit: where it is (``source``, ready for submit) and the text of its unit."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    source: Source
    text: str

    @override
    def __str__(self):
        return f"{self.source}| {self.text}"
