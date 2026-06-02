#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build FRN Chronos2 LoRA data from scratch and train the adapter."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FRN LoRA data and train Chronos2 LoRA")
    parser.add_argument("--output_dir", type=str, default="frn_results/chronos2_lora_data")
    parser.add_argument(
        "--demand_path",
        type=str,
        default="raw",
        help="raw uses FreshRetailNet-LT original train split; otherwise pass a parquet path",
    )
    parser.add_argument("--target_col", type=str, default="sale_amount")
    parser.add_argument("--official_max_per_source", type=int, default=5000)
    parser.add_argument("--force_official", action="store_true")
    parser.add_argument("--synthetic_num_series", type=int, default=20000)
    parser.add_argument("--history_length", type=int, default=84)
    parser.add_argument("--prediction_length", type=int, default=7)
    parser.add_argument("--context_length", type=int, default=35)
    parser.add_argument("--min_length", type=int, default=91)
    parser.add_argument("--train_end_date", type=str, default="2025-07-06")
    parser.add_argument("--val_end_date", type=str, default="2025-07-13")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--model_id", type=str, default="amazon/chronos-2")
    parser.add_argument("--ckpt_output_dir", type=str, default="frn_results/chronos2_lora_ckpt")
    parser.add_argument("--ckpt_name", type=str, default="frn-lora-ckpt")
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--local_files_only", action="store_true", default=True)
    parser.add_argument("--allow_download", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--skip_data_build", action="store_true", help="reuse existing train_inputs.pkl/val_inputs.pkl")
    parser.add_argument("--limit_train", type=int, default=None)
    parser.add_argument("--limit_val", type=int, default=None)
    parser.add_argument("--no_validation", action="store_true")
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print("running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)


def main() -> None:
    args = parse_args()
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    data_dir = ROOT / args.output_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_data_build:
        official_cmd = [
            sys.executable,
            str(ROOT / "demand_forecasting/chronos2/train_data_builder/download_official_univariate.py"),
            "--output_dir",
            str(data_dir / "official_univariate"),
            "--max_per_source",
            str(args.official_max_per_source),
        ]
        if args.force_official:
            official_cmd.append("--force")
        run(official_cmd)

        run([
            sys.executable,
            str(ROOT / "demand_forecasting/chronos2/train_data_builder/frn_discount_response.py"),
            "--input_path",
            args.demand_path,
            "--output_dir",
            str(data_dir),
            "--target_col",
            args.target_col,
        ])

        run([
            sys.executable,
            str(ROOT / "demand_forecasting/chronos2/train_data_builder/frn_covariate_task_generator.py"),
            "--response_path",
            str(data_dir / "frn_discount_response.parquet"),
            "--output_path",
            str(data_dir / "synthetic_inputs.pkl"),
            "--stats_path",
            str(data_dir / "synthetic_inputs_stats.json"),
            "--num_series",
            str(args.synthetic_num_series),
            "--history_length",
            str(args.history_length),
            "--prediction_length",
            str(args.prediction_length),
            "--seed",
            str(args.seed),
        ])

        run([
            sys.executable,
            str(ROOT / "demand_forecasting/chronos2/train_data_builder/frn_retail_input_builder.py"),
            "--input_path",
            args.demand_path,
            "--output_dir",
            str(data_dir),
            "--target_col",
            args.target_col,
            "--train_end_date",
            args.train_end_date,
            "--val_end_date",
            args.val_end_date,
            "--prediction_length",
            str(args.prediction_length),
            "--min_length",
            str(args.min_length),
        ])

        run([
            sys.executable,
            str(ROOT / "demand_forecasting/chronos2/train_data_builder/frn_lora_train_data_builder.py"),
            "--data_dir",
            str(data_dir),
            "--official_dir",
            str(data_dir / "official_univariate"),
            "--seed",
            str(args.seed),
        ])

    train_cmd = [
        sys.executable,
        str(ROOT / "demand_forecasting/chronos2/frn_lora_train.py"),
        "--train_inputs",
        str(data_dir / "train_inputs.pkl"),
        "--val_inputs",
        str(data_dir / "val_inputs.pkl"),
        "--model_id",
        args.model_id,
        "--output_dir",
        args.ckpt_output_dir,
        "--ckpt_name",
        args.ckpt_name,
        "--prediction_length",
        str(args.prediction_length),
        "--context_length",
        str(args.context_length),
        "--num_steps",
        str(args.num_steps),
        "--batch_size",
        str(args.batch_size),
        "--learning_rate",
        str(args.learning_rate),
        "--seed",
        str(args.seed),
    ]
    if not args.allow_download:
        train_cmd.append("--local_files_only")
    if args.deterministic:
        train_cmd.append("--deterministic")
    if args.limit_train:
        train_cmd.extend(["--limit_train", str(args.limit_train)])
    if args.limit_val:
        train_cmd.extend(["--limit_val", str(args.limit_val)])
    if args.no_validation:
        train_cmd.append("--no_validation")
    run(train_cmd)


if __name__ == "__main__":
    main()
