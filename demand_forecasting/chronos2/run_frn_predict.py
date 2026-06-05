#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run Chronos2 FRN zero-shot or LoRA prediction and summarize metrics."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

METHOD_DEMAND_PATHS = {
    "censored": None,
    "DLinear": "latent_demand_recovery/exp/demand/DLinear_demand.parquet",
    "iTransformer": "latent_demand_recovery/exp/demand/iTransformer_demand.parquet",
    "ImputeFormer": "latent_demand_recovery/exp/demand/ImputeFormer_demand.parquet",
    "TimesNet": "latent_demand_recovery/exp/demand/TimesNet_demand.parquet",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Chronos2 FRN prediction")
    parser.add_argument("--method", choices=list(METHOD_DEMAND_PATHS), default=None, help="default: run all methods")
    parser.add_argument("--lora_dir", type=str, default=None, help="LoRA adapter dir; default uses zero-shot")
    parser.add_argument("--model_id", type=str, default="amazon/chronos-2")
    parser.add_argument("--output_dir", type=str, default="frn_results")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device_map", type=str, default="cuda")
    parser.add_argument("--cutoff_date", type=str, default="2025-07-14")
    parser.add_argument("--prediction_length", type=int, default=7)
    parser.add_argument("--context_length", type=int, default=35)
    parser.add_argument("--local_files_only", action="store_true", default=True)
    parser.add_argument("--allow_download", action="store_true", help="allow model download instead of local cache only")
    parser.add_argument("--target_only", action="store_true")
    parser.add_argument("--limit_series", type=int, default=None)
    parser.add_argument("--skip_missing", action="store_true", help="skip missing recovered parquet files")
    parser.add_argument("--summary_name", type=str, default="chronos2_summary_metrics")
    return parser.parse_args()


def run_one(args: argparse.Namespace, method: str) -> None:
    demand_path = METHOD_DEMAND_PATHS[method]
    if demand_path and not (ROOT / demand_path).exists():
        msg = f"missing demand file for {method}: {demand_path}"
        if args.skip_missing:
            print("skip:", msg)
            return
        raise FileNotFoundError(msg)

    suffix = "target_only" if args.target_only else "covariates"
    if args.lora_dir:
        run_name = f"{method}_lora_{suffix}"
    else:
        run_name = f"{method}_{suffix}"

    cmd = [
        sys.executable,
        str(ROOT / "demand_forecasting/chronos2/frn_zero_shot.py"),
        "--model_id",
        args.model_id,
        "--output_dir",
        args.output_dir,
        "--run_name",
        run_name,
        "--cutoff_date",
        args.cutoff_date,
        "--prediction_length",
        str(args.prediction_length),
        "--context_length",
        str(args.context_length),
        "--batch_size",
        str(args.batch_size),
        "--device_map",
        args.device_map,
    ]
    if not args.allow_download:
        cmd.append("--local_files_only")
    if demand_path:
        cmd.extend(["--demand_path", demand_path])
    if args.lora_dir:
        cmd.extend(["--lora_dir", args.lora_dir])
    if args.target_only:
        cmd.append("--target_only")
    if args.limit_series:
        cmd.extend(["--limit_series", str(args.limit_series)])

    print("running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)


def summarize(output_dir: str, summary_name: str) -> pd.DataFrame:
    rows = []
    out = ROOT / output_dir
    for p in sorted(out.glob("*_metrics.parquet")):
        if p.name == f"{summary_name}.parquet":
            continue
        if "smoke" in p.name:
            continue
        method = p.name.replace("_metrics.parquet", "")
        df = pd.read_parquet(p).copy()
        df.insert(0, "method", method)
        rows.append(df)

    if not rows:
        raise RuntimeError(f"no metrics found in {out}")

    summary = pd.concat(rows, ignore_index=True).sort_values("wape").reset_index(drop=True)
    summary.to_csv(out / f"{summary_name}.csv", index=False)
    summary.to_parquet(out / f"{summary_name}.parquet", index=False)
    return summary


def main() -> None:
    args = parse_args()
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    methods = [args.method] if args.method else list(METHOD_DEMAND_PATHS)
    for method in methods:
        run_one(args, method)

    summary = summarize(args.output_dir, args.summary_name)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()
