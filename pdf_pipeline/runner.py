from __future__ import annotations
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime

from .config import PipelineConfig
from .metadata import parse_arxiv_from_filename
from .parsers import PyMuPDFParser, MarkItDownParser, MinerUParser, MinerUVLMParser, Parser, ParsedDoc
from .cleaning import apply_cleaning, remove_headers_footers_by_repetition, sanitize_utf8
from .writer import write_parquet


PIPELINE_VERSION = "0.1.0"


def _make_parser(name: str) -> Parser:
    if name == "pymupdf":
        return PyMuPDFParser()
    if name == "markitdown":
        return MarkItDownParser()
    if name == "mineru":
        return MinerUParser()
    if name == "mineru_vlm":
        return MinerUVLMParser()
    raise ValueError(f"Unknown parser: {name}")


def _list_pdfs(input_dir: str) -> List[str]:
    # As requested, scan the folder (non-recursive)
    return [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith(".pdf")
    ]


def _process_one(pdf_path: str, parser_name: str, cleaning_cfg: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    start = time.time()
    try:
        parser = _make_parser(parser_name)
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
            "created_at": datetime.utcnow().isoformat() + "Z",
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
            "created_at": datetime.utcnow().isoformat() + "Z",
        }
        return None, err


def run_pipeline(cfg: PipelineConfig) -> Dict[str, Any]:
    os.makedirs(os.path.dirname(cfg.output_path), exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    pdfs = _list_pdfs(cfg.input_dir)

    cleaning_cfg = {
        "unicode_nfkc": cfg.cleaning.unicode_nfkc,
        "fix_hyphenation": cfg.cleaning.fix_hyphenation,
        "remove_headers_footers": cfg.cleaning.remove_headers_footers,
    }

    num_workers = cfg.num_workers
    if num_workers is None:
        try:
            import multiprocessing as mp
            num_workers = max(1, min(8, mp.cpu_count()))
        except Exception:
            num_workers = 4

    results: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        futs = [ex.submit(_process_one, p, cfg.parser, cleaning_cfg) for p in pdfs]
        for fut in as_completed(futs):
            rec, err = fut.result()
            if rec:
                results.append(rec)
            if err:
                errors.append(err)

    # Write all outputs first (never delete parsed results)
    write_parquet(results, cfg.output_path, one_file_per_doc=cfg.one_file_per_doc)

    # Dedup pipeline: Create index marking duplicates (don't delete files)
    if cfg.dedup.enabled and results:
        try:
            from rapidfuzz import fuzz

            print(f"\n🔍 Creating dedup index for {len(results)} documents...")

            # Mark duplicates but keep all records
            for rec in results:
                rec["is_duplicate"] = False
                rec["duplicate_of"] = None

            # Group by arxiv_id (mark older versions as duplicates)
            arxiv_groups = {}
            for i, rec in enumerate(results):
                arxiv_id = rec.get("arxiv_id")
                if arxiv_id:
                    arxiv_version = rec.get("arxiv_version", "v1")
                    if arxiv_id not in arxiv_groups:
                        arxiv_groups[arxiv_id] = (i, arxiv_version)
                    else:
                        existing_idx, existing_version = arxiv_groups[arxiv_id]
                        if arxiv_version > existing_version:
                            # Mark old version as duplicate
                            results[existing_idx]["is_duplicate"] = True
                            results[existing_idx]["duplicate_of"] = rec.get("doc_id")
                            arxiv_groups[arxiv_id] = (i, arxiv_version)
                        else:
                            # Mark current as duplicate
                            rec["is_duplicate"] = True
                            rec["duplicate_of"] = results[existing_idx].get("doc_id")

            # Fuzzy dedup by title (mark similar titles as duplicates)
            if cfg.dedup.fuzzy_threshold > 0:
                for i, rec in enumerate(results):
                    if rec["is_duplicate"]:
                        continue  # Already marked

                    title = rec.get("title", "")
                    if not title:
                        continue

                    # Check similarity with previous non-duplicate titles
                    for j in range(i):
                        if results[j]["is_duplicate"]:
                            continue

                        other_title = results[j].get("title", "")
                        if not other_title:
                            continue

                        similarity = fuzz.token_set_ratio(title.lower(), other_title.lower())
                        if similarity >= cfg.dedup.fuzzy_threshold:
                            rec["is_duplicate"] = True
                            rec["duplicate_of"] = results[j].get("doc_id")
                            break

            # Count duplicates
            duplicate_count = sum(1 for rec in results if rec["is_duplicate"])
            unique_count = len(results) - duplicate_count

            print(f"✅ Marked {duplicate_count} duplicates, {unique_count} unique documents")

            # Write dedup index
            import pandas as pd
            from pathlib import Path

            index_data = []
            for rec in results:
                index_data.append({
                    "doc_id": rec.get("doc_id"),
                    "title": rec.get("title"),
                    "arxiv_id": rec.get("arxiv_id"),
                    "arxiv_version": rec.get("arxiv_version"),
                    "is_duplicate": rec["is_duplicate"],
                    "duplicate_of": rec["duplicate_of"],
                    "source_path": rec.get("source_path"),
                })

            index_df = pd.DataFrame(index_data)

            # Save index to same directory as output
            if cfg.one_file_per_doc:
                index_path = Path(cfg.output_path) / "dedup_index.parquet"
            else:
                index_path = Path(cfg.output_path).parent / "dedup_index.parquet"

            index_df.to_parquet(index_path, engine="pyarrow", compression="zstd", index=False)
            print(f"📋 Dedup index saved to: {index_path}")

        except Exception as e:
            print(f"⚠️  Dedup failed: {e}, skipping deduplication")
            import traceback
            traceback.print_exc()

    # Write logs
    try:
        import pandas as pd
        if errors:
            pd.DataFrame.from_records(errors).to_parquet("logs/errors.parquet", engine="pyarrow", compression="zstd")
        pd.DataFrame.from_records([
            {
                "total": len(pdfs),
                "success": len(results),
                "failed": len(errors),
                "parser": cfg.parser,
                "num_workers": num_workers,
                "created_at": datetime.utcnow().isoformat() + "Z",
            }
        ]).to_parquet("logs/run_metrics.parquet", engine="pyarrow", compression="zstd")
    except Exception:
        pass

    return {
        "total": len(pdfs),
        "success": len(results),
        "failed": len(errors),
        "output": cfg.output_path,
        "log_errors": "logs/errors.parquet",
    }

