from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class ParsedDoc:
    title: Optional[str]
    abstract: Optional[str]
    fulltext_markdown: str
    page_count: int


class Parser:
    name: str = "base"
    version: str = "0"

    def parse(self, pdf_path: str) -> ParsedDoc:
        raise NotImplementedError

