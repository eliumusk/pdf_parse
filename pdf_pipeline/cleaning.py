from __future__ import annotations
import re
import unicodedata
from typing import List, Optional


def unicode_nfkc(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def fix_hyphenation(text: str) -> str:
    # Join hyphenated line breaks: "exam-\nple" -> "example"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # Collapse excessive blank lines (max 2)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def remove_headers_footers_by_repetition(page_texts: List[str], min_ratio: float = 0.5) -> str:
    # Detect the first and last non-empty line of each page; remove repeated lines across many pages
    first_lines: List[str] = []
    last_lines: List[str] = []
    for t in page_texts:
        lines = [l.strip() for l in t.splitlines() if l.strip()]
        if not lines:
            first_lines.append("")
            last_lines.append("")
            continue
        first_lines.append(lines[0])
        last_lines.append(lines[-1])

    def frequent(lines: List[str]) -> Optional[str]:
        if not lines:
            return None
        from collections import Counter
        c = Counter([x for x in lines if x])
        if not c:
            return None
        item, cnt = c.most_common(1)[0]
        if cnt >= max(2, int(len(lines) * min_ratio)):
            return item
        return None

    header = frequent(first_lines)
    footer = frequent(last_lines)

    cleaned_pages: List[str] = []
    for t in page_texts:
        lines = t.splitlines()
        if header and lines and lines[0].strip() == header:
            lines = lines[1:]
        if footer and lines and lines[-1].strip() == footer:
            lines = lines[:-1]
        cleaned_pages.append("\n".join(lines))

    return "\n".join(cleaned_pages)


def sanitize_utf8(text: str) -> str:
    """Ensure text is valid UTF-8 for Parquet/Arrow.
    Replaces invalid surrogates and normalizes to NFKC.
    """
    if not isinstance(text, str):
        return text
    try:
        # Replace invalid sequences with the Unicode replacement character
        text = text.encode("utf-8", errors="replace").decode("utf-8")
    except Exception:
        # As a last resort, drop offending bytes
        text = text.encode("utf-8", errors="ignore").decode("utf-8")
    return unicodedata.normalize("NFKC", text)


def apply_cleaning(text: str, *, enable_unicode_nfkc: bool, enable_fix_hyphenation: bool) -> str:
    if enable_unicode_nfkc:
        text = unicode_nfkc(text)
    if enable_fix_hyphenation:
        text = fix_hyphenation(text)
    return text

