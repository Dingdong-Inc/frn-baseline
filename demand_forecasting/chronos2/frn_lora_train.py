#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train a Chronos2 LoRA adapter on FRN data."""

from __future__ import annotations

import argparse
import os
import pickle
import random
import sys
from pathlib import Path
from typing import Any

import torch
import numpy as np
from peft import LoraConfig

CHRONOS_DIR = Path(__file__).resolve().parent
CHRONOS_SRC = CHRONOS_DIR / "src"
sys.path.insert(0, str(CHRONOS_SRC))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Chronos2 LoRA on FRN data")
    parser.add_argument("--train_inputs", type=str, default="frn_results/chronos2_lora_data/train_inputs.pkl")
    parser.add_argument("--val_inputs", type=str, default="frn_results/chronos2_lora_data/val_inputs.pkl")
    parser.add_argument("--model_id", type=str, default="amazon/chronos-2")
    parser.add_argument("--output_dir", type=str, default="frn_results/chronos2_lora_ckpt")
    parser.add_argument("--ckpt_name", type=str, default="frn-lora-ckpt")
    parser.add_argument("--prediction_length", type=int, default=7)
    parser.add_argument("--context_length", type=int, default=35)
    parser.add_argument("--min_past", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--no_validation", action="store_true")
    parser.add_argument("--limit_train", type=int, default=None)
    parser.add_argument("--limit_val", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


def load_inputs(path: str, limit: int | None = None) -> list[dict[str, Any]]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    if limit:
        data = data[:limit]
    return data


def device_map() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main() -> None:
    args = parse_args()
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    set_seed(args.seed, args.deterministic)

    from chronos import BaseChronosPipeline

    print("loading inputs...")
    train_inputs = load_inputs(args.train_inputs, args.limit_train)
    val_inputs = None if args.no_validation else load_inputs(args.val_inputs, args.limit_val)
    print(f"train_inputs={len(train_inputs)}")
    print(f"val_inputs={0 if val_inputs is None else len(val_inputs)}")

    print("loading model...", args.model_id, "device", device_map())
    pipeline = BaseChronosPipeline.from_pretrained(
        args.model_id,
        device_map=device_map(),
        local_files_only=args.local_files_only,
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=[
            "self_attention.q",
            "self_attention.v",
            "self_attention.k",
            "self_attention.o",
            "output_patch_embedding.output_layer",
        ],
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("training params:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    pipeline.fit(
        inputs=train_inputs,
        validation_inputs=val_inputs,
        prediction_length=args.prediction_length,
        context_length=args.context_length,
        min_past=args.min_past,
        finetune_mode="lora",
        output_dir=output_dir,
        finetuned_ckpt_name=args.ckpt_name,
        learning_rate=args.learning_rate,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        lora_config=lora_config,
        remove_printer_callback=True,
        seed=args.seed,
        data_seed=args.seed,
    )

    ckpt_dir = output_dir / args.ckpt_name
    print("saved:", ckpt_dir)


if __name__ == "__main__":
    main()
