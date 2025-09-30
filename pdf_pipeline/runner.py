from __future__ import annotations
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timezone

import requests
import pandas as pd
from pathlib import Path

from .config import PipelineConfig
from .metadata import parse_arxiv_from_filename
from .parsers import PyMuPDFParser, MarkItDownParser, MinerUParser, MinerUVLMParser, Parser, ParsedDoc
from .cleaning import apply_cleaning, remove_headers_footers_by_repetition, sanitize_utf8
from .writer import write_parquet, write_single_record, append_row_to_parquet, append_error_row


PIPELINE_VERSION = "0.1.1"


def _make_parser(name: str, parser_config: dict | None = None) -> Parser:
    if name == "pymupdf":
        return PyMuPDFParser()
    if name == "markitdown":
        return MarkItDownParser()
    if name == "mineru":
        return MinerUParser()
    if name == "mineru_vlm":
        cfg = parser_config or {}
        return MinerUVLMParser(**cfg)
    raise ValueError(f"Unknown parser: {name}")


def _health_check_endpoints(api_urls: List[str], timeout: float = 3.0) -> List[str]:
    """Return only healthy endpoints by calling /health."""
    healthy: List[str] = []
    for url in api_urls:
        try:
            r = requests.get(f"{url}/health", timeout=timeout)
            if r.status_code == 200 and (r.json().get("status") == "ok"):
                healthy.append(url)
        except Exception:
            pass
    return healthy


def _version_to_int(v: Optional[str]) -> int:
    if not v:
        return 1
    try:
        if v.startswith("v"):
            return int(v[1:])
        return int(v)
    except Exception:
        return 1


import re

def _derive_abstract_if_missing(title: Optional[str], abstract: Optional[str], fulltext: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    If abstract is empty but fulltext likely contains an unlabeled abstract at the very beginning,
    extract it as the text from start until the first major section heading (e.g., INTRODUCTION),
    and return (new_abstract, new_fulltext_without_that_part).
    """
    if abstract and str(abstract).strip():
        return abstract, fulltext
    if not fulltext:
        return abstract, fulltext

    text = fulltext

    # Pattern for first section heading that typically starts the body (Introduction)
    # Supports: "1 INTRODUCTION", "I. INTRODUCTION", "# 1 INTRODUCTION", "# Introduction"
    heading_pat = re.compile(
        r"^\s{0,3}(?:#{1,3}\s*)?(?:\d+\.?|[IVXLC]+\.|)\s*(introduction|background)\b",
        re.IGNORECASE | re.MULTILINE,
    )
    m = heading_pat.search(text)

    # If no heading found, fallback: try the first markdown H1/H2 as boundary
    if not m:
        any_heading = re.compile(r"^\s{0,3}#{1,3}\s+\S+", re.MULTILINE)
        m = any_heading.search(text)

    if not m:
        # As a conservative fallback, don't invent an abstract
        return abstract, fulltext

    start = 0
    end = m.start()
    candidate = text[start:end]

    # Clean candidate: remove common metadata lines like Keywords/Index Terms
    cleaned_lines = []
    for line in candidate.splitlines():
        if re.match(r"^\s*(keywords?|index\s*terms)\s*:\s*", line, flags=re.IGNORECASE):
            continue
        cleaned_lines.append(line)
    candidate = "\n".join(cleaned_lines)

    # Collapse excessive blank lines
    candidate = re.sub(r"\n{3,}", "\n\n", candidate).strip()

    # Heuristic acceptance: require some length and at least a period or 15+ words
    if len(candidate) < 40:
        return abstract, fulltext
    if not re.search(r"[\.!?]", candidate) and len(candidate.split()) < 15:
        return abstract, fulltext

    # Remove this part from fulltext to avoid duplication
    new_fulltext = text[end:].lstrip("\n")
    return candidate, new_fulltext



def _list_pdfs(input_dir: str) -> List[str]:
    # As requested, scan the folder (non-recursive)
    return [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith(".pdf")
    ]


def _process_one(pdf_path: str, parser_name: str, cleaning_cfg: Dict[str, Any], parser_config: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    start = time.time()
    try:
        parser = _make_parser(parser_name, parser_config)
        parsed: ParsedDoc = parser.parse(pdf_path)

        # Optional header/footer removal needs page-level input; only available for PyMuPDF parser
        fulltext_md = parsed.fulltext_markdown
        if cleaning_cfg.get("remove_headers_footers", True) and parser_name == "pymupdf":
            # We cannot easily recover per-page text post-parse; instead re-open quickly for pages
            try:
                import fitz
                doc = fitz.open(pdf_path)
                pages_text = [p.get_text("text") for p in doc]
                fulltext_raw = remove_headers_footers_by_repetition(pages_text)
                fulltext_md = fulltext_raw
            except Exception:
                # best-effort; keep original
                pass

        # Apply general cleaning
        fulltext_md = apply_cleaning(
            fulltext_md,
            enable_unicode_nfkc=cleaning_cfg.get("unicode_nfkc", True),
            enable_fix_hyphenation=cleaning_cfg.get("fix_hyphenation", True),
        )

        # Metadata from filename
        ai = parse_arxiv_from_filename(pdf_path)
        arxiv_id = ai.arxiv_id if ai else None
        arxiv_version = ai.arxiv_version if ai else None
        year = ai.year if ai else None
        doc_id = ai.doc_id if ai else os.path.splitext(os.path.basename(pdf_path))[0]

        # Build record (sanitize strings to ensure valid UTF-8)
        elapsed_ms = int((time.time() - start) * 1000)
        safe_title = sanitize_utf8(parsed.title) if parsed.title is not None else None
        safe_abstract = sanitize_utf8(parsed.abstract) if parsed.abstract is not None else None
        safe_fulltext = sanitize_utf8(fulltext_md) if fulltext_md is not None else None

        # If abstract is empty but fulltext exists, try to derive abstract from the beginning
        if (safe_abstract is None or not str(safe_abstract).strip()) and (safe_fulltext and str(safe_fulltext).strip()):
            derived_abs, derived_full = _derive_abstract_if_missing(safe_title, safe_abstract, safe_fulltext)
            if derived_abs and str(derived_abs).strip():
                safe_abstract = sanitize_utf8(derived_abs)
                if derived_full:
                    safe_fulltext = sanitize_utf8(derived_full)

        record: Dict[str, Any] = {
            "doc_id": doc_id,
            "title": safe_title,
            "abstract": safe_abstract,
            "year": year,
            "fulltext": safe_fulltext,
            "arxiv_id": arxiv_id,
            "arxiv_version": arxiv_version,
            "source_path": pdf_path,
            "parser_name": parser.name,
            "parser_version": parser.version,
            "parse_time_ms": elapsed_ms,
            "page_count": parsed.page_count,
            "text_length": len(safe_fulltext) if safe_fulltext else 0,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "pipeline_version": PIPELINE_VERSION,
            "error_flags": None,
        }
        return record, None
    except Exception as e:
        elapsed_ms = int((time.time() - start) * 1000)
        err = {
            "file_path": pdf_path,
            "stage": "parse",
            "exception_type": type(e).__name__,
            "message": str(e),
            "stack_summary": traceback.format_exc(limit=5),
            "elapsed_ms": elapsed_ms,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        return None, err


def run_pipeline(cfg: PipelineConfig) -> Dict[str, Any]:
    # Prepare directories
    out_dir = Path(cfg.output_path) if cfg.one_file_per_doc else Path(cfg.output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(parents=True, exist_ok=True)

    # Health check for HTTP endpoints (only for mineru_vlm/http)
    if cfg.parser == "mineru_vlm" and isinstance(cfg.parser_config, dict):
        api_urls: List[str] = []
        pc = cfg.parser_config
        if pc.get("backend") == "http":
            if pc.get("api_urls"):
                api_urls = list(pc["api_urls"])  # copy
            elif pc.get("api_url"):
                api_urls = [pc["api_url"]]
            if api_urls:
                healthy = _health_check_endpoints(api_urls)
                if not healthy:
                    raise RuntimeError("No healthy MinerU HTTP endpoints. Please start services or fix URLs.")
                cfg.parser_config["api_urls"] = healthy
                cfg.parser_config.pop("api_url", None)
                print(f"✅ Healthy endpoints: {healthy}")

    # Resume support: load processed doc_ids from existing dedup index
    processed_doc_ids: set[str] = set()
    arxiv_heads: Dict[str, Tuple[str, int]] = {}  # arxiv_id -> (doc_id, version_int)

    index_path = (Path(cfg.output_path) / "dedup_index.parquet") if cfg.one_file_per_doc else (Path(cfg.output_path).parent / "dedup_index.parquet")
    if index_path.exists():
        try:
            df_idx = pd.read_parquet(index_path, engine="pyarrow")
            if "doc_id" in df_idx.columns:
                processed_doc_ids = set(df_idx["doc_id"].astype(str).tolist())
            if {"arxiv_id", "arxiv_version", "doc_id"}.issubset(df_idx.columns):
                for _, row in df_idx[["arxiv_id", "arxiv_version", "doc_id"]].dropna(subset=["arxiv_id"]).iterrows():
                    aid = str(row["arxiv_id"]) if pd.notna(row["arxiv_id"]) else None
                    ver = _version_to_int(str(row["arxiv_version"]) if pd.notna(row["arxiv_version"]) else None)
                    did = str(row["doc_id"]) if pd.notna(row["doc_id"]) else None
                    if aid and did:
                        head = arxiv_heads.get(aid)
                        if (head is None) or (ver > head[1]):
                            arxiv_heads[aid] = (did, ver)
        except Exception:
            pass

    # List PDFs and skip already processed doc_ids
    all_pdfs = _list_pdfs(cfg.input_dir)
    pdfs: List[str] = []
    for p in all_pdfs:
        ai = parse_arxiv_from_filename(p)
        doc_id = ai.doc_id if ai else os.path.splitext(os.path.basename(p))[0]
        if doc_id in processed_doc_ids:
            # Skip only if the per-doc parquet actually exists (resume-friendly)
            if cfg.one_file_per_doc:
                safe_doc_id = "".join(c if c.isalnum() or c in "._-" else "_" for c in doc_id)
                expected = Path(cfg.output_path) / f"{safe_doc_id}.parquet"
                if expected.exists():
                    continue
            else:
                # In single-file mode we can't check per-doc existence reliably
                continue
        pdfs.append(p)

    # Cleaning config used in workers
    cleaning_cfg = {
        "unicode_nfkc": cfg.cleaning.unicode_nfkc,
        "fix_hyphenation": cfg.cleaning.fix_hyphenation,
        "remove_headers_footers": cfg.cleaning.remove_headers_footers,
    }

    # Choose workers
    num_workers = cfg.num_workers
    if num_workers is None:
        try:
            import multiprocessing as mp
            num_workers = max(1, min(8, mp.cpu_count()))
        except Exception:
            num_workers = 4

    success_count = 0
    fail_count = 0

    # Process in parallel; stream writes per-document
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        futs = [ex.submit(_process_one, p, cfg.parser, cleaning_cfg, cfg.parser_config) for p in pdfs]
        for fut in as_completed(futs):
            rec, err = fut.result()
            if rec:
                # Stream write this document parquet
                if cfg.one_file_per_doc:
                    write_single_record(rec, Path(cfg.output_path))
                else:
                    write_parquet([rec], cfg.output_path, one_file_per_doc=False)

                # Update dedup index incrementally (version-based, fast path)
                if cfg.dedup.enabled:
                    aid = rec.get("arxiv_id")
                    ver_int = _version_to_int(rec.get("arxiv_version"))
                    is_dup = False
                    dup_of: Optional[str] = None
                    if aid:
                        head = arxiv_heads.get(aid)
                        if head is None:
                            arxiv_heads[aid] = (rec.get("doc_id"), ver_int)
                        else:
                            if ver_int <= head[1]:
                                is_dup = True
                                dup_of = head[0]
                            else:
                                arxiv_heads[aid] = (rec.get("doc_id"), ver_int)
                    index_row = {
                        "doc_id": rec.get("doc_id"),
                        "title": rec.get("title"),
                        "arxiv_id": rec.get("arxiv_id"),
                        "arxiv_version": rec.get("arxiv_version"),
                        "is_duplicate": is_dup,
                        "duplicate_of": dup_of,
                        "source_path": rec.get("source_path"),
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    }
                    append_row_to_parquet(index_path, index_row)
                success_count += 1
            if err:
                append_error_row(Path("logs/errors.parquet"), err)
                fail_count += 1

    # Metrics
    try:
        pd.DataFrame.from_records([
            {
                "total": len(all_pdfs),
                "queued": len(pdfs),
                "success": success_count,
                "failed": fail_count,
                "parser": cfg.parser,
                "num_workers": num_workers,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        ]).to_parquet("logs/run_metrics.parquet", engine="pyarrow", compression="zstd")
    except Exception:
        pass

    return {
        "total": len(all_pdfs),
        "queued": len(pdfs),
        "success": success_count,
        "failed": fail_count,
        "output": cfg.output_path,
        "log_errors": "logs/errors.parquet",
    }

