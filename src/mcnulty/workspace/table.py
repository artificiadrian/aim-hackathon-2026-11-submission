"""Tables (csv, xlsx, ods) as rows of cell text. No header guessing: rows are kept as the file shows them."""

from typing import Literal

from mcnulty.clients.arena import Source
from mcnulty.frozen import Frozen


class TableRow(Frozen):
    """csv rows carry ``line`` and ``record`` (first column); sheet rows carry ``sheet`` and ``row`` (1-based)."""

    source: Source
    cells: tuple[str, ...]
    hidden: bool = False


class Table(Frozen):
    name: str | None  # sheet name; None for csv
    hidden: bool = False
    rows: tuple[TableRow, ...]


class TableFile(Frozen):
    kind: Literal["table"] = "table"
    path: str
    tables: tuple[Table, ...]
