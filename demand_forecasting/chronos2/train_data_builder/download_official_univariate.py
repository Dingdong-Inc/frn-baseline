#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Download and cache official Chronos univariate training samples.

The first run streams samples from the official HuggingFace dataset.
Later runs reuse the local project cache when possible.
The output is jsonl, where each line can be converted directly into a
target-only Chronos2 fit input.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from datasets import load_dataset


DEFAULT_DATASET = "autogluon/chronos_datasets"
DEFAULT_CONFIGS = [
    "training_corpus_kernel_synth_1m",
    "training_corpus_tsmixup_10m",
]
DEFAULT_OUTPUT_DIR = "frn_results/chronos2_lora_data/official_univariate"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download/cache official Chronos univariate samples")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max_per_source", type=int, default=1000)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--hf_endpoint", type=str, default="https://hf-mirror.com")
    parser.add_argument("--force", action="store_true", help="ignore existing manifest and re-download")
    return parser.parse_args()


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def load_manifest(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def cache_ready(output_dir: Path, configs: list[str], max_per_source: int) -> bool:
    manifest = load_manifest(output_dir / "manifest.json")
    if not manifest:
        return False
    if int(manifest.get("max_per_source", -1)) < max_per_source:
        return False

    for cfg in configs:
        path = output_dir / f"{cfg}_sample.jsonl"
        if count_jsonl(path) < max_per_source:
            return False
    return True


def to_float_list(values: Any) -> list[float]:
    out = []
    for v in values:
        try:
            x = float(v)
        except Exception:
            continue
        if math.isfinite(x):
            out.append(x)
    return out


def normalize_record(record: dict[str, Any], source: str, idx: int) -> dict[str, Any] | None:
    target = to_float_list(record.get("target", []))
    if not target:
        return None
    return {
        "source": source,
        "source_id": record.get("id", f"{source}_{idx}"),
        "target": target,
        "past_covariates": {},
        "future_covariates": {},
    }


def download_config(dataset: str, config: str, split: str, output_path: Path, max_records: int) -> int:
    ds = load_dataset(dataset, config, split=split, streaming=True)
    written = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for idx, record in enumerate(ds):
            normalized = normalize_record(record, config, idx)
            if normalized is None:
                continue
            f.write(json.dumps(normalized, ensure_ascii=False) + "\n")
            written += 1
            if written >= max_records:
                break

    return written


def main() -> None:
    args = parse_args()
    os.environ.setdefault("HF_ENDPOINT", args.hf_endpoint)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    configs = list(args.configs)

    if not args.force and cache_ready(output_dir, configs, args.max_per_source):
        manifest = load_manifest(output_dir / "manifest.json")
        print("cache ready:", output_dir)
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return

    sources = []
    for config in configs:
        out_path = output_dir / f"{config}_sample.jsonl"
        print(f"downloading {args.dataset}/{config} -> {out_path}")
        count = download_config(args.dataset, config, args.split, out_path, args.max_per_source)
        sources.append({"config": config, "path": str(out_path), "sample_cnt": count})
        print(f"saved {count} records")

    manifest = {
        "dataset": args.dataset,
        "split": args.split,
        "max_per_source": args.max_per_source,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sources": sources,
        "total_sample_cnt": int(sum(s["sample_cnt"] for s in sources)),
    }
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print("saved manifest:", manifest_path)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
