from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Optional


def write_parquet(records: List[Dict[str, Any]], output_path: str, one_file_per_doc: bool = False) -> None:
    """
    Write parsed documents to Parquet file(s)

    Args:
        records: List of document records
        output_path: Output path (file or directory)
        one_file_per_doc: If True, save each document as a separate file
    """
    if not records:
        # create empty file with schema
        import pandas as pd
        df = pd.DataFrame(columns=[
            "doc_id", "title", "abstract", "year", "fulltext",
            "arxiv_id", "arxiv_version", "source_path", "parser_name", "parser_version",
            "parse_time_ms", "page_count", "text_length",
            "created_at", "pipeline_version", "error_flags",
        ])
        if not one_file_per_doc:
            df.to_parquet(output_path, engine="pyarrow", compression="zstd")
        return

    import pandas as pd

    # Column order
    cols = [
        "doc_id", "title", "abstract", "year", "fulltext",
        "arxiv_id", "arxiv_version", "source_path", "parser_name", "parser_version",
        "parse_time_ms", "page_count", "text_length",
        "created_at", "pipeline_version", "error_flags",
    ]

    if one_file_per_doc:
        # Save each document as a separate file
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        for rec in records:
            write_single_record(rec, output_dir)
    else:
        # Save all documents in one file (original behavior)
        df = pd.DataFrame.from_records(records)
        cols_existing = [c for c in cols if c in df.columns]
        df = df[cols_existing]

        # Write with zstd when available
        try:
            df.to_parquet(output_path, engine="pyarrow", compression="zstd", index=False)
        except Exception:
            df.to_parquet(output_path, engine="pyarrow", compression="snappy", index=False)


def write_single_record(rec: Dict[str, Any], output_dir: Path) -> Path:
    """Write a single record to <output_dir>/<doc_id>.parquet and return the file path."""
    import pandas as pd

    # Column order
    cols = [
        "doc_id", "title", "abstract", "year", "fulltext",
        "arxiv_id", "arxiv_version", "source_path", "parser_name", "parser_version",
        "parse_time_ms", "page_count", "text_length",
        "created_at", "pipeline_version", "error_flags",
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    doc_id = rec.get("doc_id", "unknown")
    safe_doc_id = "".join(c if c.isalnum() or c in "._-" else "_" for c in doc_id)

    df = pd.DataFrame([rec])
    cols_existing = [c for c in cols if c in df.columns]
    df = df[cols_existing]

    output_file = output_dir / f"{safe_doc_id}.parquet"
    try:
        df.to_parquet(output_file, engine="pyarrow", compression="zstd", index=False)
    except Exception:
        df.to_parquet(output_file, engine="pyarrow", compression="snappy", index=False)
    return output_file


def append_row_to_parquet(index_path: Path, row: Dict[str, Any]) -> None:
    """
    Append a single row to a parquet file by read-modify-write.
    NOTE: For large-scale runs a dataset/appendable format is better, but this
    keeps correctness and simplicity per the current requirement.
    """
    import pandas as pd

    index_path.parent.mkdir(parents=True, exist_ok=True)
    if index_path.exists():
        try:
            df_old = pd.read_parquet(index_path, engine="pyarrow")
        except Exception:
            df_old = pd.DataFrame()
        # Upsert by doc_id: drop existing rows with same doc_id then append
        if not df_old.empty and "doc_id" in df_old.columns and ("doc_id" in row):
            df_old = df_old[df_old["doc_id"].astype(str) != str(row["doc_id"])].copy()
        df_new = pd.concat([df_old, pd.DataFrame([row])], ignore_index=True)
    else:
        df_new = pd.DataFrame([row])

    try:
        df_new.to_parquet(index_path, engine="pyarrow", compression="zstd", index=False)
    except Exception:
        df_new.to_parquet(index_path, engine="pyarrow", compression="snappy", index=False)


def append_error_row(errors_path: Path, row: Dict[str, Any]) -> None:
    """Append one error row to errors parquet (read-modify-write)."""
    append_row_to_parquet(errors_path, row)
