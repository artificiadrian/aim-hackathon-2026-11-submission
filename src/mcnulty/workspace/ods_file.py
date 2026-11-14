"""OpenDocument spreadsheets: every sheet, every row as shown, hidden rows and sheets marked."""

from xml.etree.ElementTree import Element

from mcnulty.clients.arena import Source
from mcnulty.workspace.odf import attr, content, paragraphs_text, tag
from mcnulty.workspace.table import Table, TableFile, TableRow

MAX_REPEAT = 1_000  # repeated empty rows and columns at the end of a sheet can repeat thousands of times


def row_cells(row: Element):
    cells: list[str] = []
    for cell in row.iter(tag("table", "table-cell")):
        repeat = int(attr(cell, "table", "number-columns-repeated") or 1)
        cells.extend([paragraphs_text(cell)] * min(repeat, MAX_REPEAT))
    while cells and not cells[-1]:
        cells.pop()
    return tuple(cells)


def parse_ods(path: str, data: bytes):
    root = content(data)
    hidden_styles = {
        attr(style, "style", "name")
        for style in root.iter(tag("style", "style"))
        for props in style.iter(tag("style", "table-properties"))
        if attr(props, "table", "display") == "false"
    }
    tables = []
    for table in root.iter(tag("table", "table")):
        name = attr(table, "table", "name") or ""
        rows: list[TableRow] = []
        for row in table.iter(tag("table", "table-row")):
            cells = row_cells(row)
            hidden = attr(row, "table", "visibility") == "collapse"
            for _ in range(min(int(attr(row, "table", "number-rows-repeated") or 1), MAX_REPEAT)):
                source = Source(path=path, sheet=name, row=len(rows) + 1)
                rows.append(TableRow(source=source, cells=cells, hidden=hidden))
        while rows and not any(rows[-1].cells):
            rows.pop()
        sheet_hidden = (
            attr(table, "table", "visibility") == "collapse" or attr(table, "table", "style-name") in hidden_styles
        )
        tables.append(Table(name=name, hidden=sheet_hidden, rows=tuple(rows)))
    return TableFile(path=path, tables=tuple(tables))
