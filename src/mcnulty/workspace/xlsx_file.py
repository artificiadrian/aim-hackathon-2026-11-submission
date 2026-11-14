"""Excel workbooks: every sheet, every row as shown (title rows included), hidden rows and sheets marked."""

import io
import warnings

import openpyxl

from mcnulty.arena import Source
from mcnulty.workspace.table import Table, TableFile, TableRow


def cell_text(value: object):
    return "" if value is None else str(value)


def parse_xlsx(path: str, data: bytes):
    with warnings.catch_warnings():
        # Harmless: the generated workbooks have no style sheet and openpyxl applies its default.
        warnings.filterwarnings("ignore", message="Workbook contains no default style", category=UserWarning)
        book = openpyxl.load_workbook(io.BytesIO(data), data_only=True)  # data_only: formula results, not formulas
    tables = []
    for sheet in book.worksheets:
        rows = tuple(
            TableRow(
                source=Source(path=path, sheet=sheet.title, row=n),
                cells=tuple(cell_text(c.value) for c in cells),
                hidden=bool(sheet.row_dimensions[n].hidden),
            )
            for n, cells in enumerate(sheet.iter_rows(), 1)
        )
        tables.append(Table(name=sheet.title, hidden=sheet.sheet_state != "visible", rows=rows))
    return TableFile(path=path, tables=tuple(tables))
