from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any


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

        print(f"\n💾 Saving {len(records)} documents to {output_dir}/")

        for rec in records:
            doc_id = rec.get("doc_id", "unknown")
            # Sanitize filename
            safe_doc_id = "".join(c if c.isalnum() or c in "._-" else "_" for c in doc_id)

            # Create single-row DataFrame
            df = pd.DataFrame([rec])
            cols_existing = [c for c in cols if c in df.columns]
            df = df[cols_existing]

            # Write to file
            output_file = output_dir / f"{safe_doc_id}.parquet"
            try:
                df.to_parquet(output_file, engine="pyarrow", compression="zstd", index=False)
            except Exception:
                df.to_parquet(output_file, engine="pyarrow", compression="snappy", index=False)

        print(f"✅ Saved {len(records)} files to {output_dir}/")

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

