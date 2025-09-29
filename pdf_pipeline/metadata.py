from __future__ import annotations
import os
import re
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class ArxivInfo:
    arxiv_id: str  # without version
    arxiv_version: Optional[str]  # like 'v2'
    doc_id: str  # e.g., '2406.19549v2'
    year: Optional[int]


_ARXIV_NEW_RE = re.compile(r"^(\d{2})(\d{2})\.(\d{5})(v\d+)?$")


def parse_arxiv_from_filename(path: str) -> Optional[ArxivInfo]:
    name = os.path.splitext(os.path.basename(path))[0]
    m = _ARXIV_NEW_RE.match(name)
    if not m:
        return None
    yy, mm, num, ver = m.groups()
    year = 2000 + int(yy)
    arxiv_id = f"{yy}{mm}.{num}"
    doc_id = f"{arxiv_id}{ver or ''}"
    return ArxivInfo(arxiv_id=arxiv_id, arxiv_version=ver, doc_id=doc_id, year=year)

