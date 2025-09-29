from __future__ import annotations
import re
from typing import Optional
from .base import Parser, ParsedDoc


def _extract_title_from_markdown(md: str) -> Optional[str]:
    for line in md.splitlines()[:50]:
        if line.startswith("# "):
            t = line[2:].strip()
            if t:
                return t
    # fallback: first non-empty line
    for line in md.splitlines():
        s = line.strip()
        if s:
            return s
    return None


def _extract_abstract_from_markdown(md: str) -> Optional[str]:
    # Look for 'Abstract' section in markdown
    m = re.search(r"(?is)^(?:##|#)?\s*abstract\s*\n+(.+?)(?:\n\s*\n|\n(?:##|#)\s+|\Z)", md, re.MULTILINE)
    if m:
        return m.group(1).strip()
    return None


class MarkItDownParser(Parser):
    name = "markitdown"
    version = "2"

    def parse(self, pdf_path: str) -> ParsedDoc:
        try:
            from markitdown import MarkItDown
        except Exception as e:
            raise RuntimeError("'markitdown' package is required for parser 'markitdown'. Please install 'markitdown'.") from e

        md = MarkItDown()
        result = md.convert(pdf_path)
        md_text: str = getattr(result, "text_content", "") or ""

        title = _extract_title_from_markdown(md_text)
        abstract = _extract_abstract_from_markdown(md_text)

        return ParsedDoc(
            title=title,
            abstract=abstract,
            fulltext_markdown=md_text.strip(),
            page_count=-1,  # markitdown does not expose page count
        )

