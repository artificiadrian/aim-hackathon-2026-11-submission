"""CSV: each row is cited by its physical line or by its record id (first column)."""

import csv
import io

from mcnulty.clients.arena import Source
from mcnulty.workspace.table import Table, TableFile, TableRow


def parse_csv(path: str, data: bytes):
    reader = csv.reader(io.StringIO(data.decode("utf-8"), newline=""))
    rows = []
    for cells in reader:
        line = reader.line_num  # the physical line the row ends on
        record = cells[0] if cells and line > 1 else None  # line 1 is the header
        rows.append(TableRow(source=Source(path=path, line=line, record=record), cells=tuple(cells)))
    return TableFile(path=path, tables=(Table(name=None, rows=tuple(rows)),))
