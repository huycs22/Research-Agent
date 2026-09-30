"""Small, text-only document parsers used by the research library."""

from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path

from langchain_core.documents import Document

PARSER_VERSION = 1
SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md", ".html"}


class _TextHTMLParser(HTMLParser):
    blocked = {"script", "style", "noscript", "svg", "nav", "header", "footer"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag.casefold() in self.blocked:
            self.depth += 1

    def handle_endtag(self, tag):
        if tag.casefold() in self.blocked and self.depth:
            self.depth -= 1

    def handle_data(self, data):
        if not self.depth:
            self.parts.append(data)

    def text(self) -> str:
        return re.sub(r"[ \t]+", " ", " ".join(self.parts)).replace(" \n", "\n").strip()


def _clean(text: str) -> str:
    return text.replace("\x00", "").replace("\ufffd", "")


def _pdf_pages(path: Path, page_size_limit: int) -> list[str]:
    """Prefer PyMuPDF and fall back to pypdf without importing PaperQA readers."""
    try:
        import pymupdf

        with pymupdf.open(path) as pdf:
            pages = [_clean(page.get_text("text", sort=True)) for page in pdf]
    except (ImportError, RuntimeError, ValueError):
        from pypdf import PdfReader

        pages = [_clean(page.extract_text() or "") for page in PdfReader(path).pages]
    for number, text in enumerate(pages, 1):
        if len(text) > page_size_limit:
            raise ValueError(
                f"Page {number} in {path.name} exceeds the {page_size_limit}-character limit."
            )
    return pages


def parse_document(path: Path, document_id: str, *, page_size_limit: int = 100_000) -> list[Document]:
    suffix = path.suffix.casefold()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported document type: {suffix}")
    if suffix == ".pdf":
        texts = _pdf_pages(path, page_size_limit)
    else:
        text = path.read_text("utf-8", errors="ignore")
        if suffix == ".html":
            parser = _TextHTMLParser()
            parser.feed(text)
            text = parser.text()
        texts = [text]
    return [
        Document(
            page_content=text,
            metadata={
                "document_id": document_id,
                "filename": path.name,
                "title": path.stem,
                "page": str(number),
                "citation": path.stem,
                "parser": f"text-only-v{PARSER_VERSION}",
            },
        )
        for number, text in enumerate(texts, 1)
        if text.strip()
    ]

