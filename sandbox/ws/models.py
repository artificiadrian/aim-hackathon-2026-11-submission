"""What the helpers return: pydantic models, so they print, serialize and go into submit(...) as they are."""

from typing import ClassVar

from pydantic import BaseModel, ConfigDict


class Source(BaseModel):
    """A citation: the path plus the locator the platform grades for that file type (docs/ARENA_DOCS.md)."""

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
        fields = self.model_dump(exclude={"path"}, exclude_none=True)
        return " ".join(f"{k}={v}" for k, v in fields.items()) or "file"


class Unit(BaseModel):
    """One citable unit of a file: a line, a csv record, a sheet row, a paragraph, a page."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    source: Source
    text: str
    cells: tuple[str, ...] = ()  # the cells of a table row, in column order


class Hit(BaseModel):
    """A search hit: where it is (``source``, ready for submit) and the text of its unit."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    source: Source
    text: str

    def __str__(self):
        return f"{self.source.path} {self.source.where}| {self.text}"
