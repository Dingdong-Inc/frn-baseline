#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build Chronos2 LoRA train/validation inputs from real FRN retail data."""

from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_INPUT = "raw"
DEFAULT_OUTPUT_DIR = "frn_results/chronos2_lora_data"
COVARIATE_COLS = ["discount", "holiday_flag", "day_of_week", "precpt", "avg_temperature"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FRN retail Chronos2 LoRA inputs")
    parser.add_argument("--input_path", type=str, default=DEFAULT_INPUT)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target_col", type=str, default="sale_amount")
    parser.add_argument("--train_end_date", type=str, default="2025-07-06")
    parser.add_argument("--val_end_date", type=str, default="2025-07-13")
    parser.add_argument("--prediction_length", type=int, default=7)
    parser.add_argument("--min_length", type=int, default=91)
    parser.add_argument("--limit_series", type=int, default=None)
    return parser.parse_args()


def read_data(input_path: str, target_col: str) -> pd.DataFrame:
    columns = [
        "store_id",
        "product_id",
        "dt",
        target_col,
        "discount",
        "holiday_flag",
        "precpt",
        "avg_temperature",
    ]
    if input_path == "raw":
        from datasets import load_dataset

        dataset = load_dataset("Dingdong-Inc/FreshRetailNet-LT")
        df = dataset["train"].to_pandas()[columns]
    else:
        df = pd.read_parquet(input_path, columns=columns)
    df = df.rename(columns={target_col: "target"})
    df["dt"] = pd.to_datetime(df["dt"])
    df["day_of_week"] = df["dt"].dt.dayofweek.astype(np.float32)

    for col in ["target", "discount", "holiday_flag", "precpt", "avg_temperature"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fill missing values with simple, controlled defaults.
    global_temp_median = float(df["avg_temperature"].median()) if df["avg_temperature"].notna().any() else 20.0
    df["target"] = df["target"].fillna(0.0).clip(lower=0.0)
    df["discount"] = df["discount"].fillna(1.0).clip(lower=0.3, upper=1.0)
    df["holiday_flag"] = df["holiday_flag"].fillna(0.0)
    df["precpt"] = df["precpt"].fillna(0.0).clip(lower=0.0)
    df["avg_temperature"] = df["avg_temperature"].fillna(global_temp_median)

    for col in ["target", *COVARIATE_COLS]:
        df[col] = df[col].astype(np.float32)

    df = (
        df.sort_values(["store_id", "product_id", "dt"])
        .drop_duplicates(subset=["store_id", "product_id", "dt"], keep="last")
        .reset_index(drop=True)
    )
    return df


def pack_covariates(g: pd.DataFrame) -> dict[str, np.ndarray]:
    return {col: g[col].to_numpy(dtype=np.float32) for col in COVARIATE_COLS}


def build_inputs(
    df: pd.DataFrame,
    end_date: str,
    min_length: int,
    limit_series: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    end = pd.to_datetime(end_date)
    part = df[df["dt"] <= end].copy()
    part = part.sort_values(["store_id", "product_id", "dt"])

    inputs: list[dict[str, Any]] = []
    lengths = []
    dropped_short = 0
    dropped_all_zero = 0

    for _, g in part.groupby(["store_id", "product_id"], sort=False):
        if len(g) < min_length:
            dropped_short += 1
            continue

        target = g["target"].to_numpy(dtype=np.float32)
        if np.all(target <= 0):
            dropped_all_zero += 1
            continue

        inputs.append(
            {
                "target": target,
                "past_covariates": pack_covariates(g),
                "future_covariates": {col: None for col in COVARIATE_COLS},
            }
        )
        lengths.append(len(g))

        if limit_series and len(inputs) >= limit_series:
            break

    length_arr = np.asarray(lengths, dtype=np.float32) if lengths else np.asarray([], dtype=np.float32)
    stats = {
        "end_date": end_date,
        "num_inputs": int(len(inputs)),
        "min_length_required": int(min_length),
        "dropped_short": int(dropped_short),
        "dropped_all_zero": int(dropped_all_zero),
        "length_min": int(length_arr.min()) if len(length_arr) else 0,
        "length_mean": float(length_arr.mean()) if len(length_arr) else 0.0,
        "length_median": float(np.median(length_arr)) if len(length_arr) else 0.0,
        "length_max": int(length_arr.max()) if len(length_arr) else 0,
    }
    return inputs, stats


def save_pickle(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = read_data(args.input_path, args.target_col)
    train_inputs, train_stats = build_inputs(
        df=df,
        end_date=args.train_end_date,
        min_length=args.min_length,
        limit_series=args.limit_series,
    )
    val_inputs, val_stats = build_inputs(
        df=df,
        end_date=args.val_end_date,
        min_length=args.min_length,
        limit_series=args.limit_series,
    )

    train_path = output_dir / "retail_train_inputs.pkl"
    val_path = output_dir / "retail_val_inputs.pkl"
    stats_path = output_dir / "retail_inputs_stats.json"
    save_pickle(train_inputs, train_path)
    save_pickle(val_inputs, val_path)

    stats = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_path": args.input_path,
        "target_col": args.target_col,
        "prediction_length": int(args.prediction_length),
        "covariates": COVARIATE_COLS,
        "train": train_stats,
        "val": val_stats,
    }
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    if train_inputs:
        first = train_inputs[0]
        print("first train target shape:", np.asarray(first["target"]).shape)
        print("first train cov shapes:", {k: np.asarray(v).shape for k, v in first["past_covariates"].items()})
    print("saved:", train_path)
    print("saved:", val_path)
    print("saved:", stats_path)


if __name__ == "__main__":
    main()
