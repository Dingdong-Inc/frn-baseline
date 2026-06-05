#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Merge official univariate, real FRN retail, and synthetic FRN data into LoRA inputs."""

from __future__ import annotations

import argparse
import json
import pickle
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_DATA_DIR = "frn_results/chronos2_lora_data"
DEFAULT_OFFICIAL_DIR = "frn_results/chronos2_lora_data/official_univariate"
COVARIATE_COLS = ["discount", "holiday_flag", "day_of_week", "precpt", "avg_temperature"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build final Chronos2 LoRA train/val inputs")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--official_dir", type=str, default=DEFAULT_OFFICIAL_DIR)
    parser.add_argument("--retail_train_path", type=str, default=None)
    parser.add_argument("--retail_val_path", type=str, default=None)
    parser.add_argument("--synthetic_path", type=str, default=None)
    parser.add_argument("--output_train_path", type=str, default=None)
    parser.add_argument("--output_val_path", type=str, default=None)
    parser.add_argument("--stats_path", type=str, default=None)
    parser.add_argument("--official_n", type=int, default=-1, help="-1 means use all available")
    parser.add_argument("--retail_n", type=int, default=-1, help="-1 means use all available")
    parser.add_argument("--synthetic_n", type=int, default=-1, help="-1 means use all available")
    parser.add_argument("--val_n", type=int, default=-1, help="-1 means use all retail val inputs")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def save_pickle(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)


def to_float32_array(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    return arr.astype(np.float32)


def load_official_inputs(official_dir: Path) -> list[dict[str, Any]]:
    inputs: list[dict[str, Any]] = []
    for path in sorted(official_dir.glob("*_sample.jsonl")):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                target = to_float32_array(rec.get("target", []))
                if len(target) == 0:
                    continue
                inputs.append(
                    {
                        "target": target,
                        "past_covariates": {},
                        "future_covariates": {},
                    }
                )
    return inputs


def sample_inputs(inputs: list[dict[str, Any]], n: int, rng: random.Random) -> list[dict[str, Any]]:
    if n < 0 or n >= len(inputs):
        sampled = list(inputs)
    else:
        sampled = rng.sample(inputs, n)
    return sampled


def input_length_stats(inputs: list[dict[str, Any]]) -> dict[str, Any]:
    if not inputs:
        return {"count": 0, "length_min": 0, "length_mean": 0.0, "length_median": 0.0, "length_max": 0}
    lengths = np.asarray([len(x["target"]) for x in inputs], dtype=np.float32)
    return {
        "count": int(len(inputs)),
        "length_min": int(lengths.min()),
        "length_mean": float(lengths.mean()),
        "length_median": float(np.median(lengths)),
        "length_max": int(lengths.max()),
    }


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    data_dir = Path(args.data_dir)
    official_dir = Path(args.official_dir)

    retail_train_path = Path(args.retail_train_path) if args.retail_train_path else data_dir / "retail_train_inputs.pkl"
    retail_val_path = Path(args.retail_val_path) if args.retail_val_path else data_dir / "retail_val_inputs.pkl"
    synthetic_path = Path(args.synthetic_path) if args.synthetic_path else data_dir / "synthetic_inputs.pkl"
    output_train_path = Path(args.output_train_path) if args.output_train_path else data_dir / "train_inputs.pkl"
    output_val_path = Path(args.output_val_path) if args.output_val_path else data_dir / "val_inputs.pkl"
    stats_path = Path(args.stats_path) if args.stats_path else data_dir / "build_stats.json"

    official_inputs = load_official_inputs(official_dir)
    retail_train_inputs = load_pickle(retail_train_path)
    retail_val_inputs = load_pickle(retail_val_path)
    synthetic_inputs = load_pickle(synthetic_path)

    official_sample = sample_inputs(official_inputs, args.official_n, rng)
    retail_sample = sample_inputs(retail_train_inputs, args.retail_n, rng)
    synthetic_sample = sample_inputs(synthetic_inputs, args.synthetic_n, rng)
    val_sample = sample_inputs(retail_val_inputs, args.val_n, rng)

    train_inputs = official_sample + retail_sample + synthetic_sample
    rng.shuffle(train_inputs)

    save_pickle(train_inputs, output_train_path)
    save_pickle(val_sample, output_val_path)

    stats = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "seed": int(args.seed),
        "paths": {
            "official_dir": str(official_dir),
            "retail_train_path": str(retail_train_path),
            "retail_val_path": str(retail_val_path),
            "synthetic_path": str(synthetic_path),
            "output_train_path": str(output_train_path),
            "output_val_path": str(output_val_path),
        },
        "available": {
            "official": input_length_stats(official_inputs),
            "retail_train": input_length_stats(retail_train_inputs),
            "retail_val": input_length_stats(retail_val_inputs),
            "synthetic": input_length_stats(synthetic_inputs),
        },
        "selected": {
            "official": input_length_stats(official_sample),
            "retail_train": input_length_stats(retail_sample),
            "synthetic": input_length_stats(synthetic_sample),
            "train_total": input_length_stats(train_inputs),
            "val": input_length_stats(val_sample),
        },
    }
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    if train_inputs:
        first = train_inputs[0]
        print("first train keys:", first.keys())
        print("first train target shape:", np.asarray(first["target"]).shape)
        print("first train cov keys:", list(first.get("past_covariates", {}).keys()))
    print("saved:", output_train_path)
    print("saved:", output_val_path)
    print("saved:", stats_path)


if __name__ == "__main__":
    main()
