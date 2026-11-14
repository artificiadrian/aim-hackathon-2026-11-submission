"""Shared OpenDocument (ods, odt) reading: ``content.xml`` with a hardened XML parser."""

import io
import zipfile

from lxml import etree

# Untrusted files: no entity expansion, no DTD, no network, no huge trees.
PARSER = etree.XMLParser(resolve_entities=False, load_dtd=False, no_network=True, huge_tree=False)
NAMESPACES = {
    "table": "urn:oasis:names:tc:opendocument:xmlns:table:1.0",
    "text": "urn:oasis:names:tc:opendocument:xmlns:text:1.0",
    "office": "urn:oasis:names:tc:opendocument:xmlns:office:1.0",
    "style": "urn:oasis:names:tc:opendocument:xmlns:style:1.0",
}


def content(data: bytes):
    return etree.fromstring(zipfile.ZipFile(io.BytesIO(data)).read("content.xml"), PARSER)


def tag(namespace: str, name: str):
    return f"{{{NAMESPACES[namespace]}}}{name}"


def attr(element: etree._Element, namespace: str, name: str):
    return element.get(tag(namespace, name))


def paragraphs_text(element: etree._Element):
    return "\n".join("".join(p.itertext()) for p in element.iter(tag("text", "p")))
