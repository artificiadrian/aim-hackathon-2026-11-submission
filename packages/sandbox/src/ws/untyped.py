"""Readers that use libraries without type information (odfpy; pandas without its separate stubs). The checks for
unknown types are off in this module only; the results are typed ``Unit`` models like every other reader."""

# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportAttributeAccessIssue=false, reportAny=false

from ws.models import Source, Unit


def ods_units(path: str):
    import pandas as pd

    found: list[Unit] = []
    for name, frame in pd.read_excel(path, engine="odf", header=None, sheet_name=None).items():
        for index, row in frame.iterrows():
            cells = tuple("" if pd.isna(v) else str(v) for v in row)
            if any(cells):
                source = Source(path=path, sheet=str(name), row=int(str(index)) + 1)
                found.append(Unit(source=source, text=" | ".join(c for c in cells if c), cells=cells))
    return found


def odt_units(path: str):
    from odf import teletype
    from odf.namespaces import OFFICENS, TEXTNS
    from odf.opendocument import load

    paragraphs: list[str] = []
    stack: list[object] = [load(path).text]
    while stack:  # document order; comments (office:annotation) are not body text
        node = stack.pop()
        name = getattr(node, "qname", None)
        if name in ((TEXTNS, "p"), (TEXTNS, "h")):
            paragraphs.append(teletype.extractText(node))
        elif name != (OFFICENS, "annotation"):
            stack.extend(reversed(getattr(node, "childNodes", [])))
    texts = [t for t in paragraphs if t.strip()]
    return [Unit(source=Source(path=path, paragraph=n), text=t) for n, t in enumerate(texts, 1)]
