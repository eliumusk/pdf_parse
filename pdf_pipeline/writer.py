from __future__ import annotations
from typing import List, Dict, Any


def write_parquet(records: List[Dict[str, Any]], output_path: str) -> None:
    if not records:
        # create empty file with schema? Keep it simple: write empty DataFrame
        import pandas as pd
        df = pd.DataFrame(columns=[
            "doc_id", "title", "abstract", "year", "fulltext",
            "arxiv_id", "arxiv_version", "source_path", "parser_name", "parser_version",
            "parse_time_ms", "page_count", "text_length",
            "created_at", "pipeline_version", "error_flags",
        ])
        df.to_parquet(output_path, engine="pyarrow", compression="zstd")
        return

    import pandas as pd

    df = pd.DataFrame.from_records(records)

    # Reorder columns if present
    cols = [
        "doc_id", "title", "abstract", "year", "fulltext",
        "arxiv_id", "arxiv_version", "source_path", "parser_name", "parser_version",
        "parse_time_ms", "page_count", "text_length",
        "created_at", "pipeline_version", "error_flags",
    ]
    cols_existing = [c for c in cols if c in df.columns]
    df = df[cols_existing]

    # Write with zstd when available
    try:
        df.to_parquet(output_path, engine="pyarrow", compression="zstd")
    except Exception:
        df.to_parquet(output_path, engine="pyarrow", compression="snappy")

