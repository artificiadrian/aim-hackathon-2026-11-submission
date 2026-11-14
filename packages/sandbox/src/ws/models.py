"""What the helpers return: pydantic models, so they print, serialize and go into submit(...) as they are."""

from typing import ClassVar, override

from pydantic import BaseModel, ConfigDict


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
    def where(self):
        """The locator as text, e.g. ``sheet=Alerts row=58``; ``file`` for a path-only citation."""
        fields = {
            "line": self.line,
            "record": self.record,
            "sheet": self.sheet,
            "row": self.row,
            "paragraph": self.paragraph,
            "page": self.page,
        }
        return " ".join(f"{k}={v}" for k, v in fields.items() if v is not None) or "file"


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
        return f"{self.source.path} {self.source.where}| {self.text}"
