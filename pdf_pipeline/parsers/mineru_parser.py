from __future__ import annotations
import os
import re
import tempfile
from pathlib import Path
from typing import Optional

from .base import Parser, ParsedDoc
from dotenv import load_dotenv
load_dotenv()


def _extract_title_from_markdown(md: str) -> Optional[str]:
    for line in md.splitlines()[:50]:
        if line.startswith("# "):
            t = line[2:].strip()
            if t:
                return t
    for line in md.splitlines():
        s = line.strip()
        if s:
            return s
    return None


def _extract_abstract_from_markdown(md: str) -> Optional[str]:
    m = re.search(r"(?is)^(?:##|#)?\s*abstract\s*\n+(.+?)(?:\n\s*\n|\n(?:##|#)\s+|\Z)", md, re.MULTILINE)
    if m:
        return m.group(1).strip()
    return None


class MinerUParser(Parser):
    name = "mineru"
    version = "2"

    def __init__(self, *, language: str = "en", enable_formula: bool = True, enable_table: bool = True):
        self.language = language
        self.enable_formula = enable_formula
        self.enable_table = enable_table

    def parse(self, pdf_path: str) -> ParsedDoc:
        # Import mineru only when needed to reduce import time for other parsers
        from mineru.cli.common import convert_pdf_bytes_to_bytes_by_pypdfium2, prepare_env, read_fn
        from mineru.data.data_reader_writer import FileBasedDataWriter
        from mineru.backend.pipeline.pipeline_analyze import doc_analyze as pipeline_doc_analyze
        from mineru.backend.pipeline.model_json_to_middle_json import result_to_middle_json as pipeline_result_to_middle_json
        from mineru.backend.pipeline.pipeline_middle_json_mkcontent import union_make as pipeline_union_make
        from mineru.utils.enum_class import MakeMode

        # Read PDF bytes and normalize pages (as in demo)
        pdf_bytes = read_fn(pdf_path)
        pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_bytes, 0, None)

        # Run pipeline analyze for a single document
        infer_results, all_image_lists, all_pdf_docs, lang_list, ocr_enabled_list = pipeline_doc_analyze(
            [pdf_bytes], [self.language], parse_method="auto", formula_enable=self.enable_formula, table_enable=self.enable_table
        )

        model_list = infer_results[0]
        images_list = all_image_lists[0]
        pdf_doc = all_pdf_docs[0]
        _lang = lang_list[0]
        _ocr_enable = ocr_enabled_list[0]

        # Use a temporary directory to store any intermediate images written by mineru
        pdf_file_name = Path(pdf_path).stem
        with tempfile.TemporaryDirectory(prefix="mineru_tmp_") as tmpdir:
            local_image_dir, local_md_dir = prepare_env(tmpdir, pdf_file_name, "auto")
            image_writer = FileBasedDataWriter(local_image_dir)

            middle_json = pipeline_result_to_middle_json(
                model_list, images_list, pdf_doc, image_writer, _lang, _ocr_enable, self.enable_formula
            )

            pdf_info = middle_json["pdf_info"]
            image_dir_basename = os.path.basename(local_image_dir)
            md_content_str = pipeline_union_make(pdf_info, MakeMode.MM_MD, image_dir_basename)

        title = _extract_title_from_markdown(md_content_str)
        abstract = _extract_abstract_from_markdown(md_content_str)

        return ParsedDoc(
            title=title,
            abstract=abstract,
            fulltext_markdown=md_content_str.strip(),
            page_count=-1,
        )

