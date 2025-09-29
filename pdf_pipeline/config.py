from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import yaml


@dataclass
class CleaningConfig:
    unicode_nfkc: bool = True
    fix_hyphenation: bool = True
    remove_headers_footers: bool = True


@dataclass
class DedupConfig:
    enabled: bool = False
    fuzzy_threshold: int = 90  # rapidfuzz token_set_ratio


@dataclass
class PipelineConfig:
    parser: str = "pymupdf"  # one of: pymupdf | markitdown | mineru
    input_dir: str = "test/"
    output_path: str = "docs/docs.parquet"
    num_workers: Optional[int] = None  # None -> auto

    # Behavior
    allow_network: bool = False  # default: offline

    # Modules
    cleaning: CleaningConfig = field(default_factory=CleaningConfig)
    dedup: DedupConfig = field(default_factory=DedupConfig)


def load_config(path: str) -> PipelineConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    def _merge(dc, d):
        for k, v in d.items():
            if hasattr(dc, k):
                setattr(dc, k, v)
        return dc

    cfg = PipelineConfig()
    # Top-level simple fields
    for k in ["parser", "input_dir", "output_path", "num_workers", "allow_network"]:
        if k in data:
            setattr(cfg, k, data[k])

    # Nested configs
    if isinstance(data.get("cleaning"), dict):
        cfg.cleaning = _merge(CleaningConfig(), data["cleaning"])  # type: ignore
    if isinstance(data.get("dedup"), dict):
        cfg.dedup = _merge(DedupConfig(), data["dedup"])  # type: ignore

    return cfg

