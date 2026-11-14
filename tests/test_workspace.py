"""Each parser on a small generated file: citations, hidden parts and comments as the format shows them."""

import io
import zipfile

import docx
import openpyxl

from mcnulty.workspace.csv_file import parse_csv
from mcnulty.workspace.docx_file import parse_docx
from mcnulty.workspace.jsonl_file import parse_jsonl
from mcnulty.workspace.ods_file import parse_ods
from mcnulty.workspace.odt_file import parse_odt
from mcnulty.workspace.parse import UnreadableFile, parse_file
from mcnulty.workspace.text_file import parse_text
from mcnulty.workspace.unknown_file import BinaryFile
from mcnulty.workspace.xlsx_file import parse_xlsx

ODF_HEAD = (
    '<office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
    'xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0" '
    'xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0" '
    'xmlns:style="urn:oasis:names:tc:opendocument:xmlns:style:1.0">'
)


def odf_zip(body: str):
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as z:
        z.writestr("content.xml", ODF_HEAD + body + "</office:document-content>")
    return out.getvalue()


def test_text_counts_blank_lines_and_not_the_final_newline():
    parsed = parse_text("/a.md", b"# Title\n\nline 3\n")
    assert [(line.source.line, line.text) for line in parsed.lines] == [(1, "# Title"), (2, ""), (3, "line 3")]


def test_jsonl_keeps_invalid_lines_as_text():
    parsed = parse_jsonl("/log.jsonl", b'{"a": 1}\nnot json\n')
    assert parsed.lines[0].value == {"a": 1}
    assert parsed.lines[1].value is None
    assert parsed.lines[1].text == "not json"
    assert parsed.lines[1].source.line == 2


def test_csv_rows_carry_line_and_record():
    parsed = parse_csv("/t.csv", b"id,name\nK-01,Ada\nK-02,Bo\n")
    rows = parsed.tables[0].rows
    assert rows[0].source.record is None  # header
    assert (rows[2].source.line, rows[2].source.record, rows[2].cells) == (3, "K-02", ("K-02", "Bo"))


def test_xlsx_keeps_title_rows_and_marks_hidden_rows_and_sheets():
    book = openpyxl.Workbook()
    sheet = book.active
    assert sheet is not None
    sheet.title = "transfers"
    sheet.append(["Register title"])
    sheet.append([])
    sheet.append(["tx_id", "records_n"])
    sheet.append(["TX-1", 5])
    sheet.row_dimensions[4].hidden = True
    secret = book.create_sheet("_review")
    secret.sheet_state = "hidden"
    secret.append(["note"])
    out = io.BytesIO()
    book.save(out)

    parsed = parse_xlsx("/r.xlsx", out.getvalue())
    transfers, review = parsed.tables
    row = transfers.rows[3]
    assert (row.source.sheet, row.source.row, row.cells, row.hidden) == ("transfers", 4, ("TX-1", "5"), True)
    assert not transfers.hidden
    assert review.hidden


def test_ods_counts_rows_and_marks_hidden_rows_and_sheets():
    body = (
        "<office:automatic-styles>"
        '<style:style style:name="ta2"><style:table-properties table:display="false"/></style:style>'
        "</office:automatic-styles><office:body><office:spreadsheet>"
        '<table:table table:name="transfers">'
        "<table:table-row><table:table-cell><text:p>title</text:p></table:table-cell></table:table-row>"
        '<table:table-row table:visibility="collapse"><table:table-cell><text:p>TX-1</text:p></table:table-cell>'
        '<table:table-cell table:number-columns-repeated="3"/></table:table-row>'
        '<table:table-row table:number-rows-repeated="500"><table:table-cell/></table:table-row>'
        "</table:table>"
        '<table:table table:name="_review" table:style-name="ta2">'
        "<table:table-row><table:table-cell><text:p>note</text:p></table:table-cell></table:table-row>"
        "</table:table></office:spreadsheet></office:body>"
    )
    parsed = parse_ods("/r.ods", odf_zip(body))
    transfers, review = parsed.tables
    assert [(r.source.row, r.cells, r.hidden) for r in transfers.rows] == [(1, ("title",), False), (2, ("TX-1",), True)]
    assert not transfers.hidden
    assert review.hidden


def test_docx_numbers_non_empty_paragraphs_and_keeps_comments_apart():
    document = docx.Document()
    document.add_heading("Agreement", level=0)
    document.add_paragraph("")
    body = document.add_paragraph("7.3 Notify within 48 hours.")
    document.add_comment(body.runs[0], text="reviewer note", author="r")
    out = io.BytesIO()
    document.save(out)

    parsed = parse_docx("/a.docx", out.getvalue())
    assert [(p.source.paragraph, p.text) for p in parsed.paragraphs] == [
        (1, "Agreement"),
        (2, "7.3 Notify within 48 hours."),
    ]
    assert parsed.comments == ("reviewer note",)


def test_odt_counts_p_and_h_and_keeps_annotations_apart():
    body = (
        "<office:body><office:text>"
        "<text:h>Handover</text:h><text:p></text:p>"
        "<text:p>Shift notes<office:annotation><text:p>reviewer note</text:p></office:annotation></text:p>"
        "</office:text></office:body>"
    )
    parsed = parse_odt("/h.odt", odf_zip(body))
    assert [(p.source.paragraph, p.text) for p in parsed.paragraphs] == [(1, "Handover"), (2, "Shift notes")]
    assert parsed.comments == ("reviewer note",)


def test_unknown_binary_and_broken_files_are_data_not_crashes():
    assert isinstance(parse_file("/blob.bin", b"\xff\xfe\x00"), BinaryFile)
    broken = parse_file("/broken.xlsx", b"not a zip")
    assert isinstance(broken, UnreadableFile)
    assert broken.error.type.endswith("BadZipFile")
