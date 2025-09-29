from __future__ import annotations
import re
from typing import List, Optional
from .base import Parser, ParsedDoc


def _extract_title_from_text(text: str) -> Optional[str]:
    # Try to find a prominent title before Abstract or first numbered heading
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for i, line in enumerate(lines[:15]):
        if re.match(r"(?i)^abstract\\b", line):
            break
        if re.match(r"^\\d+(?:[.\\d]*)\\s+", line):
            # likely first section
            break
        if len(line) > 5 and len(line) < 200:
            return line
    return None


def _extract_abstract_from_text(text: str) -> Optional[str]:
    # Look for Abstract section
    m = re.search(r"(?is)\babstract\b\s*:?\s*(.+?)(?:\n\s*\n|\n\s*1\s+intro|\n\s*1\.|\n##\s*introduction|\Z)", text)
    if m:
        return m.group(1).strip()
    return None


class PyMuPDFParser(Parser):
    name = "pymupdf"
    version = "1"

    def parse(self, pdf_path: str) -> ParsedDoc:
        try:
            import fitz  # PyMuPDF
        except Exception as e:
            raise RuntimeError("PyMuPDF (fitz) is required for parser 'pymupdf'. Please install 'pymupdf'.") from e

        doc = fitz.open(pdf_path)
        pages_text: List[str] = []
        for page in doc:
            # Use simple text extraction; layout-aware methods are heavier
            pages_text.append(page.get_text("text"))
        raw_text = "\n".join(pages_text)

        meta_title = (doc.metadata or {}).get("title") or None
        title = meta_title or _extract_title_from_text(pages_text[0] if pages_text else raw_text)
        abstract = _extract_abstract_from_text(raw_text)

        # Keep as lightly structured Markdown: convert numbered headings to markdown-style
        md_lines: List[str] = []
        for line in raw_text.splitlines():
            if re.match(r"^\s*\d+(?:[.\d]*)\s+\S", line):
                md_lines.append("# " + line.strip())
            else:
                md_lines.append(line.rstrip())
        fulltext_md = "\n".join(md_lines).strip()

        return ParsedDoc(
            title=title,
            abstract=abstract,
            fulltext_markdown=fulltext_md,
            page_count=len(doc),
        )

