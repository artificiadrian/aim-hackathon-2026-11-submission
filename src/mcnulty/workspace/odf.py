"""Shared OpenDocument (ods, odt) reading: ``content.xml`` with the standard library parser.

Safe for untrusted files: Python 3.12 bundles expat >= 2.6, which blocks billion-laughs and quadratic-blowup entity
attacks, and ElementTree never loads external entities or DTDs.
"""

import io
import zipfile
from xml.etree import ElementTree
from xml.etree.ElementTree import Element

NAMESPACES = {
    "table": "urn:oasis:names:tc:opendocument:xmlns:table:1.0",
    "text": "urn:oasis:names:tc:opendocument:xmlns:text:1.0",
    "office": "urn:oasis:names:tc:opendocument:xmlns:office:1.0",
    "style": "urn:oasis:names:tc:opendocument:xmlns:style:1.0",
}


def content(data: bytes):
    xml = zipfile.ZipFile(io.BytesIO(data)).read("content.xml")
    return ElementTree.fromstring(xml)  # noqa: S314  (see module docstring)


def tag(namespace: str, name: str):
    return f"{{{NAMESPACES[namespace]}}}{name}"


def attr(element: Element, namespace: str, name: str):
    return element.get(tag(namespace, name))


def paragraphs_text(element: Element):
    return "\n".join("".join(p.itertext()) for p in element.iter(tag("text", "p")))
