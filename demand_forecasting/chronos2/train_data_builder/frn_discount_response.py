#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fit FRN discount response parameters at management-group granularity.

The current FRN data has no explicit price field, so this only fits
management_group_id-level parameters.
The objective keeps the logit_func from the original task:
    t = exp(b * (discount - 1))
    uplift = a * t / (t + a - 1)
"""

from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore")

DEFAULT_INPUT = "raw"
DEFAULT_OUTPUT_DIR = "frn_results/chronos2_lora_data"
SAMPLE_LIMIT = 1000
GLOBAL_GROUP = "__global__"


def logit_func(x, a, b):
    t = np.exp(b * (x - 1))
    return (a * t) / (t + a - 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit FRN discount response by management_group_id")
    parser.add_argument("--input_path", type=str, default=DEFAULT_INPUT)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target_col", type=str, default="sale_amount")
    parser.add_argument("--sample_limit", type=int, default=SAMPLE_LIMIT)
    parser.add_argument("--nonpromo_threshold", type=float, default=0.98)
    parser.add_argument("--min_nonpromo_cnt", type=int, default=3)
    parser.add_argument("--min_discount", type=float, default=0.3)
    parser.add_argument("--max_discount", type=float, default=1.0)
    parser.add_argument("--clip_min", type=float, default=0.2)
    parser.add_argument("--clip_max", type=float, default=5.0)
    parser.add_argument("--bin_width", type=float, default=0.02)
    parser.add_argument("--max_rows", type=int, default=None, help="debug only: sample first N rows after load")
    return parser.parse_args()


def read_data(input_path: str, target_col: str, max_rows: int | None = None) -> pd.DataFrame:
    columns = ["management_group_id", "store_id", "product_id", "dt", "discount", target_col]
    if input_path == "raw":
        from datasets import load_dataset

        dataset = load_dataset("Dingdong-Inc/FreshRetailNet-LT")
        df = dataset["train"].to_pandas()[columns]
    else:
        df = pd.read_parquet(input_path, columns=columns)
    if max_rows:
        df = df.head(max_rows).copy()
    df = df.rename(columns={target_col: "target"})

    df["discount"] = pd.to_numeric(df["discount"], errors="coerce")
    df["target"] = pd.to_numeric(df["target"], errors="coerce")
    df = df.dropna(subset=["management_group_id", "store_id", "product_id", "discount", "target"])
    return df


def add_base_demand(df: pd.DataFrame, nonpromo_threshold: float, min_nonpromo_cnt: int) -> pd.DataFrame:
    keys = ["store_id", "product_id"]

    all_base = df.groupby(keys, sort=False)["target"].median().rename("base_all")
    nonpromo = df[df["discount"] >= nonpromo_threshold]
    nonpromo_stats = nonpromo.groupby(keys, sort=False)["target"].agg(base_nonpromo="median", nonpromo_cnt="count")

    out = df.merge(all_base, on=keys, how="left").merge(nonpromo_stats, on=keys, how="left")
    out["nonpromo_cnt"] = out["nonpromo_cnt"].fillna(0)
    use_nonpromo = out["nonpromo_cnt"] >= min_nonpromo_cnt
    out["base_demand"] = np.where(use_nonpromo, out["base_nonpromo"], out["base_all"])
    return out


def prepare_fit_data(args: argparse.Namespace) -> pd.DataFrame:
    df = read_data(args.input_path, args.target_col, args.max_rows)
    df = add_base_demand(df, args.nonpromo_threshold, args.min_nonpromo_cnt)

    df = df[
        (df["discount"] >= args.min_discount)
        & (df["discount"] <= args.max_discount)
        & (df["target"] > 0)
        & (df["base_demand"] > 0)
    ].copy()
    df["breakout_rate"] = (df["target"] / df["base_demand"]).clip(args.clip_min, args.clip_max)
    df["sigma"] = 1 / np.clip(df["base_demand"].to_numpy(dtype=float), 0.25, 5.0)
    df["management_group_id"] = df["management_group_id"].astype(str)
    return df


def cal_param(g: pd.DataFrame) -> tuple[float, float] | None:
    x = g["discount"].astype(float).to_numpy()
    y = g["breakout_rate"].astype(float).to_numpy()
    sigma = g["sigma"].astype(float).to_numpy()
    try:
        popt, _ = curve_fit(
            logit_func,
            x,
            y,
            sigma=sigma,
            maxfev=500000,
            bounds=([-10, -10], [40, 1]),
        )
        a, b = float(np.round(popt[0], 4)), float(np.round(popt[1], 4))
        if not np.isfinite(a) or not np.isfinite(b):
            return None
        return a, b
    except Exception:
        return None


def metrics(g: pd.DataFrame, a: float, b: float) -> dict[str, float]:
    y = g["breakout_rate"].astype(float).to_numpy()
    pred = logit_func(g["discount"].astype(float).to_numpy(), a, b)
    err = np.abs(pred - y)
    weight = 1 / g["sigma"].astype(float).to_numpy()
    return {
        "fit_mae": float(np.mean(err)),
        "fit_wmae": float(np.average(err, weights=weight)),
    }


def summarize_group(
    group_id: str,
    g: pd.DataFrame,
    params: tuple[float, float],
    used_global_fallback: bool,
) -> dict[str, Any]:
    a, b = params
    m = metrics(g, a, b)
    return {
        "management_group_id": group_id,
        "logit_params_w": f"{a},{b}",
        "a": a,
        "b": b,
        "cnt": int(len(g)),
        "discount_min": float(g["discount"].min()),
        "discount_max": float(g["discount"].max()),
        "breakout_rate_mean": float(g["breakout_rate"].mean()),
        "breakout_rate_median": float(g["breakout_rate"].median()),
        "fit_mae": m["fit_mae"],
        "fit_wmae": m["fit_wmae"],
        "used_global_fallback": bool(used_global_fallback),
    }


def build_response(df: pd.DataFrame, sample_limit: int) -> pd.DataFrame:
    global_params = cal_param(df)
    if global_params is None:
        global_params = (1.0, 0.0)

    rows = [summarize_group(GLOBAL_GROUP, df, global_params, False)]
    for group_id, g in df.groupby("management_group_id", sort=True):
        if len(g) >= sample_limit:
            params = cal_param(g)
            used_fallback = params is None
            params = params or global_params
        else:
            params = global_params
            used_fallback = True
        rows.append(summarize_group(group_id, g, params, used_fallback))

    return pd.DataFrame(rows)


def build_bins(df: pd.DataFrame, bin_width: float) -> pd.DataFrame:
    df = df.copy()
    df["discount_bin"] = (np.round(df["discount"] / bin_width) * bin_width).round(4)
    return (
        df.groupby(["management_group_id", "discount_bin"], sort=True)["breakout_rate"]
        .agg(cnt="count", breakout_rate_mean="mean", breakout_rate_median="median")
        .reset_index()
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = prepare_fit_data(args)
    response = build_response(df, args.sample_limit)
    bins = build_bins(df, args.bin_width)

    response_csv = output_dir / "frn_discount_response.csv"
    response_parquet = output_dir / "frn_discount_response.parquet"
    bins_csv = output_dir / "frn_discount_response_bins.csv"
    meta_path = output_dir / "frn_discount_response_meta.json"

    response.to_csv(response_csv, index=False)
    response.to_parquet(response_parquet, index=False)
    bins.to_csv(bins_csv, index=False)

    meta = {
        "input_path": args.input_path,
        "target_col": args.target_col,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "fit_sample_cnt": int(len(df)),
        "group_cnt": int(response["management_group_id"].nunique() - 1),
        "sample_limit": int(args.sample_limit),
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(response.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("saved:", response_csv)
    print("saved:", response_parquet)
    print("saved:", bins_csv)
    print("saved:", meta_path)


if __name__ == "__main__":
    main()
