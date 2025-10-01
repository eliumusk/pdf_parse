import re
import argparse
from pathlib import Path
import pandas as pd

FAIL_PATTERNS = [
    r"Extraction failed", r"Bad Gateway", r"502", r"Failed to call MinerU API",
]

def is_bad_record(df: pd.DataFrame) -> bool:
    title = str(df.get("title", [""])[0] if "title" in df.columns and len(df) else "")
    fulltext = str(df.get("fulltext", [""])[0] if "fulltext" in df.columns and len(df) else "")
    text = title + "\n" + fulltext
    return any(re.search(p, text, re.IGNORECASE) for p in FAIL_PATTERNS)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="docs/mineru_vlm 目录")
    ap.add_argument("--delete", action="store_true", help="直接删除，不进回收站")
    ap.add_argument("--dry-run", action="store_true", help="只打印，不执行")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    index_path = root / "dedup_index.parquet"
    trash = root / "trash"
    trash.mkdir(parents=True, exist_ok=True)

    candidates = []
    for p in root.glob("*.parquet"):
        if p.name == "dedup_index.parquet":
            continue
        try:
            df = pd.read_parquet(p, engine="pyarrow", columns=["doc_id","title","fulltext"])
        except Exception:
            # 读失败也当坏样本
            candidates.append((p, None))
            continue
        if is_bad_record(df):
            candidates.append((p, df))

    print(f"Found {len(candidates)} bad parquet file(s).")
    for p, _ in candidates:
        print(" -", p.name)

    if args.dry_run:
        return

    # 处理 dedup_index：删除对应 doc_id 行
    if index_path.exists() and candidates:
        idx = pd.read_parquet(index_path, engine="pyarrow")
        if "doc_id" in idx.columns and len(idx):
            bad_ids = set()
            for p, df in candidates:
                if df is not None and "doc_id" in df.columns:
                    bad_ids.add(str(df["doc_id"].iloc[0]))
                else:
                    bad_ids.add(p.stem)  # 文件名即 doc_id（已被清洗）
            new_idx = idx[~idx["doc_id"].astype(str).isin(bad_ids)].copy()
            new_idx.to_parquet(index_path, engine="pyarrow", compression="zstd", index=False)
            print(f"Updated index: removed {len(idx)-len(new_idx)} rows from dedup_index.parquet")

    # 移动到 trash 或直接删除
    for p, _ in candidates:
        if args.delete:
            p.unlink(missing_ok=True)
        else:
            p.rename(trash / p.name)
    print("Done.")

if __name__ == "__main__":
    main()