#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate FRN-style synthetic Chronos2 training samples.

Flow:
1. Reuse generate_base_univariate() to create a base univariate series.
2. Generate FRN-style covariates: discount, holiday_flag, day_of_week,
   precpt, and avg_temperature.
3. Apply the logit formula to target values using the discount response
   parameters a,b for each management_group_id.
4. Write a list[dict] pickle that can be passed directly to Chronos2 fit().
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Allow direct script execution to import covariate_task_generator.py from this directory.
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from covariate_task_generator import generate_base_univariate, minmax_scale


DEFAULT_RESPONSE_PATH = "frn_results/chronos2_lora_data/frn_discount_response.parquet"
DEFAULT_OUTPUT_PATH = "frn_results/chronos2_lora_data/synthetic_inputs.pkl"
DEFAULT_STATS_PATH = "frn_results/chronos2_lora_data/synthetic_inputs_stats.json"
COVARIATE_COLS = ["discount", "holiday_flag", "day_of_week", "precpt", "avg_temperature"]


def logit_func(x: np.ndarray, a: float, b: float) -> np.ndarray:
    t = np.exp(b * (x - 1.0))
    return (a * t) / (t + a - 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate FRN-style synthetic Chronos2 inputs")
    parser.add_argument("--response_path", type=str, default=DEFAULT_RESPONSE_PATH)
    parser.add_argument("--output_path", type=str, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--stats_path", type=str, default=DEFAULT_STATS_PATH)
    parser.add_argument("--num_series", type=int, default=20000)
    parser.add_argument("--history_length", type=int, default=84)
    parser.add_argument("--prediction_length", type=int, default=7)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max_kernels", type=int, default=5)
    parser.add_argument("--clip_target_max", type=float, default=80.0)
    parser.add_argument("--include_debug_fields", action="store_true")
    return parser.parse_args()


def load_response_table(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    required = {"management_group_id", "a", "b"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"response table missing columns: {missing}")

    df = df.copy()
    df["management_group_id"] = df["management_group_id"].astype(str)
    df["a"] = pd.to_numeric(df["a"], errors="coerce")
    df["b"] = pd.to_numeric(df["b"], errors="coerce")
    df = df.dropna(subset=["a", "b"])
    if df.empty:
        raise ValueError("empty response table")
    return df


def sample_group_params(response: pd.DataFrame, rng: np.random.Generator) -> tuple[str, float, float]:
    groups = response[response["management_group_id"] != "__global__"].copy()
    if groups.empty:
        groups = response.copy()

    if "cnt" in groups.columns and groups["cnt"].sum() > 0:
        weights = groups["cnt"].to_numpy(dtype=float)
        weights = weights / weights.sum()
        idx = int(rng.choice(np.arange(len(groups)), p=weights))
    else:
        idx = int(rng.integers(0, len(groups)))

    row = groups.iloc[idx]
    return str(row["management_group_id"]), float(row["a"]), float(row["b"])


def generate_discount(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate blocky discounts close to FRN: mostly 1, with some medium/large discounts."""
    discount = np.ones(n, dtype=np.float32)
    i = 0
    while i < n:
        block = int(rng.integers(2, 8))
        p = float(rng.random())
        if p < 0.58:
            value = 1.0
        elif p < 0.83:
            value = float(rng.uniform(0.90, 0.99))
        elif p < 0.96:
            value = float(rng.uniform(0.75, 0.90))
        else:
            value = float(rng.uniform(0.30, 0.75))
        discount[i : min(n, i + block)] = value
        i += block
    return discount.astype(np.float32)


def generate_precpt(n: int, rng: np.random.Generator) -> np.ndarray:
    precpt = np.zeros(n, dtype=np.float32)
    i = 0
    while i < n:
        if rng.random() < 0.22:
            width = int(rng.integers(1, 5))
            regime = rng.choice(["light", "medium", "heavy"], p=[0.78, 0.17, 0.05])
            if regime == "light":
                center = float(rng.uniform(0.2, 5.0))
            elif regime == "medium":
                center = float(rng.uniform(5.0, 15.0))
            else:
                center = float(rng.uniform(15.0, 30.0))
            for j in range(i, min(n, i + width)):
                precpt[j] = np.clip(center + rng.normal(0, 0.15 * max(center, 1.0)), 0, 35)
            i += width
        else:
            i += 1
    return precpt.astype(np.float32)


def generate_temperature(n: int, start_date: pd.Timestamp, rng: np.random.Generator) -> np.ndarray:
    day_of_year = np.array([(start_date + pd.Timedelta(days=i)).dayofyear for i in range(n)])
    phase = 2 * np.pi * (day_of_year - 30) / 365.0
    seasonal = 18 + 11 * np.sin(phase)
    noise = rng.normal(0.0, 2.0, size=n)
    return np.clip(seasonal + noise, -5, 38).astype(np.float32)


def generate_covariates(n: int, rng: np.random.Generator) -> tuple[pd.DatetimeIndex, dict[str, np.ndarray]]:
    start = pd.Timestamp("2023-01-01") + pd.Timedelta(days=int(rng.integers(0, 365)))
    dates = pd.date_range(start, periods=n, freq="D")
    day_of_week = dates.dayofweek.to_numpy(dtype=np.float32)

    holiday_flag = np.zeros(n, dtype=np.float32)
    # Treat weekends and a few random events as holidays/promotion days.
    holiday_flag[np.isin(day_of_week, [5, 6])] = 1.0
    random_holidays = rng.choice(np.arange(n), size=max(1, n // 30), replace=False)
    holiday_flag[random_holidays] = 1.0

    cov = {
        "discount": generate_discount(n, rng),
        "holiday_flag": holiday_flag.astype(np.float32),
        "day_of_week": day_of_week.astype(np.float32),
        "precpt": generate_precpt(n, rng),
        "avg_temperature": generate_temperature(n, start, rng),
    }
    return dates, cov


def build_sales(
    base_signal: np.ndarray,
    cov: dict[str, np.ndarray],
    a: float,
    b: float,
    rng: np.random.Generator,
    clip_target_max: float,
) -> np.ndarray:
    n = len(base_signal)
    # Keep the base sales range small, matching FRN per-store per-product daily sales.
    base_scaled = minmax_scale(base_signal, 0.0, 1.0)
    level = float(rng.uniform(0.8, 8.0))
    amplitude = float(rng.uniform(0.3, 4.0))
    base_sales = level + amplitude * base_scaled

    day_of_week = cov["day_of_week"]
    weekend_effect = np.where(np.isin(day_of_week, [5, 6]), rng.uniform(0.95, 1.20), 1.0)
    holiday_effect = 1.0 + cov["holiday_flag"] * float(rng.uniform(0.00, 0.18))

    temp = cov["avg_temperature"]
    temp_effect = 1.0 + 0.004 * (temp - np.nanmean(temp))
    temp_effect = np.clip(temp_effect, 0.85, 1.15)

    # Precipitation may affect fresh retail demand in either direction; sample the sign.
    rain_sign = float(rng.choice([-1.0, 1.0], p=[0.45, 0.55]))
    rain_effect = 1.0 + rain_sign * 0.025 * np.sqrt(cov["precpt"])
    rain_effect = np.clip(rain_effect, 0.75, 1.35)

    discount_uplift = logit_func(cov["discount"], a, b)
    discount_uplift = np.nan_to_num(discount_uplift, nan=1.0, posinf=3.0, neginf=0.2)
    discount_uplift = np.clip(discount_uplift, 0.2, 5.0)

    noise = rng.lognormal(mean=0.0, sigma=0.12, size=n)
    sales = base_sales * weekend_effect * holiday_effect * temp_effect * rain_effect * discount_uplift * noise
    sales = np.clip(sales, 0.0, clip_target_max)
    return sales.astype(np.float32)


def build_sample(
    idx: int,
    total_length: int,
    response: pd.DataFrame,
    rng: np.random.Generator,
    max_kernels: int,
    clip_target_max: float,
    include_debug_fields: bool,
) -> dict[str, Any]:
    group_id, a, b = sample_group_params(response, rng)
    base, generator_name = generate_base_univariate(
        length=total_length,
        max_kernels=max_kernels,
        generator_type=None,
        seed=int(rng.integers(0, 2**31 - 1)),
    )
    _, cov = generate_covariates(total_length, rng)
    target = build_sales(base, cov, a, b, rng, clip_target_max)

    sample = {
        "target": target,
        "past_covariates": {col: cov[col].astype(np.float32) for col in COVARIATE_COLS},
        "future_covariates": {col: None for col in COVARIATE_COLS},
    }
    if include_debug_fields:
        sample["_debug"] = {
            "task_id": f"frn_synth_{idx:06d}",
            "management_group_id": group_id,
            "a": a,
            "b": b,
            "base_generator": generator_name,
        }
    return sample


def generate_inputs(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    random.seed(args.seed)
    np.random.seed(args.seed)

    response = load_response_table(args.response_path)
    total_length = args.history_length + args.prediction_length
    rng = np.random.default_rng(args.seed)

    inputs = []
    for i in range(args.num_series):
        inputs.append(
            build_sample(
                idx=i,
                total_length=total_length,
                response=response,
                rng=rng,
                max_kernels=args.max_kernels,
                clip_target_max=args.clip_target_max,
                include_debug_fields=args.include_debug_fields,
            )
        )
        if (i + 1) % 1000 == 0:
            print(f"generated {i + 1}/{args.num_series}")

    targets = np.concatenate([np.asarray(x["target"], dtype=np.float32) for x in inputs])
    stats = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "response_path": args.response_path,
        "num_series": int(args.num_series),
        "history_length": int(args.history_length),
        "prediction_length": int(args.prediction_length),
        "total_length": int(total_length),
        "seed": int(args.seed),
        "covariates": COVARIATE_COLS,
        "target_min": float(np.min(targets)),
        "target_mean": float(np.mean(targets)),
        "target_median": float(np.median(targets)),
        "target_max": float(np.max(targets)),
    }
    return inputs, stats


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_path)
    stats_path = Path(args.stats_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    inputs, stats = generate_inputs(args)
    with output_path.open("wb") as f:
        pickle.dump(inputs, f)
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("first input keys:", inputs[0].keys() if inputs else None)
    if inputs:
        print("first target shape:", np.asarray(inputs[0]["target"]).shape)
        print("first cov shapes:", {k: np.asarray(v).shape for k, v in inputs[0]["past_covariates"].items()})
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print("saved:", output_path)
    print("saved:", stats_path)


if __name__ == "__main__":
    main()
