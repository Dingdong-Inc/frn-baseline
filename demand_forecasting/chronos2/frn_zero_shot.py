import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset


ROOT = Path(__file__).resolve().parents[2]
CHRONOS_SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(CHRONOS_SRC))

COVARIATE_COLS = ["discount", "holiday_flag", "day_of_week", "precpt", "avg_temperature"]


def load_frn_data(demand_path=None):
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    dataset = load_dataset("Dingdong-Inc/FreshRetailNet-LT")
    raw_train_df = dataset["train"].to_pandas()
    eval_df = dataset["eval"].to_pandas()

    if demand_path:
        train_df = pd.read_parquet(demand_path)
        target_col = "sale_amount_pred"
    else:
        train_df = raw_train_df
        target_col = "sale_amount"

    return train_df, eval_df, raw_train_df, target_col


def build_target_only_inputs(
    train_df,
    eval_df,
    target_col,
    cutoff_date="2025-07-14",
    prediction_length=7,
    context_length=None,
    limit_series=None,
):
    train_df = train_df.copy()
    eval_df = eval_df.copy()
    train_df["dt"] = pd.to_datetime(train_df["dt"])
    eval_df["dt"] = pd.to_datetime(eval_df["dt"])
    cutoff = pd.to_datetime(cutoff_date)

    eval_items = eval_df[["store_id", "product_id"]].drop_duplicates()
    train_df = train_df.merge(eval_items, on=["store_id", "product_id"])
    train_df = train_df[train_df["dt"] < cutoff]
    train_df = train_df.sort_values(["store_id", "product_id", "dt"])

    eval_dates = pd.date_range(cutoff, periods=prediction_length)
    eval_need = eval_df[eval_df["dt"].isin(eval_dates)]
    full_eval_items = (
        eval_need.groupby(["store_id", "product_id"])["dt"]
        .nunique()
        .reset_index(name="n_dates")
        .query("n_dates == @prediction_length")[["store_id", "product_id"]]
    )
    train_df = train_df.merge(full_eval_items, on=["store_id", "product_id"])

    chronos_inputs = []
    index_rows = []
    dropped_short = 0
    dropped_all_nan = 0

    for keys, group in train_df.groupby(["store_id", "product_id"], sort=False):
        target = pd.to_numeric(group[target_col], errors="coerce").to_numpy(dtype=np.float32)
        if context_length:
            target = target[-context_length:]
        if len(target) == 0:
            dropped_short += 1
            continue
        if np.all(np.isnan(target)):
            dropped_all_nan += 1
            continue

        chronos_inputs.append({"target": target})
        index_rows.append({"store_id": keys[0], "product_id": keys[1]})

        if limit_series and len(chronos_inputs) >= limit_series:
            break

    index_df = pd.DataFrame(index_rows)
    stats = {
        "target_col": target_col,
        "cutoff_date": cutoff_date,
        "prediction_length": prediction_length,
        "context_length": context_length,
        "num_series": len(chronos_inputs),
        "dropped_short": dropped_short,
        "dropped_all_nan": dropped_all_nan,
    }
    return chronos_inputs, index_df, eval_df, stats


def add_day_of_week(df):
    df = df.copy()
    df["day_of_week"] = pd.to_datetime(df["dt"]).dt.dayofweek.astype(np.float32)
    return df


def prepare_covariates(df):
    df = add_day_of_week(df)
    for col in COVARIATE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float32)
    return df


def pack_covariates(df):
    return {col: df[col].to_numpy(dtype=np.float32) for col in COVARIATE_COLS}


def build_covariate_inputs(
    train_df,
    eval_df,
    target_col,
    cutoff_date="2025-07-14",
    prediction_length=7,
    context_length=None,
    limit_series=None,
):
    train_df = prepare_covariates(train_df)
    eval_df = prepare_covariates(eval_df)
    train_df["dt"] = pd.to_datetime(train_df["dt"])
    eval_df["dt"] = pd.to_datetime(eval_df["dt"])
    cutoff = pd.to_datetime(cutoff_date)

    eval_dates = pd.date_range(cutoff, periods=prediction_length)
    future_df = eval_df[eval_df["dt"].isin(eval_dates)]
    full_eval_items = (
        future_df.groupby(["store_id", "product_id"])["dt"]
        .nunique()
        .reset_index(name="n_dates")
        .query("n_dates == @prediction_length")[["store_id", "product_id"]]
    )

    train_df = train_df.merge(full_eval_items, on=["store_id", "product_id"])
    train_df = train_df[train_df["dt"] < cutoff]
    train_df = train_df.sort_values(["store_id", "product_id", "dt"])
    future_df = future_df.merge(full_eval_items, on=["store_id", "product_id"])
    future_df = future_df.sort_values(["store_id", "product_id", "dt"])

    chronos_inputs = []
    index_rows = []
    dropped_short = 0
    dropped_all_nan = 0
    dropped_future = 0

    future_groups = {
        keys: group for keys, group in future_df.groupby(["store_id", "product_id"], sort=False)
    }

    for keys, context in train_df.groupby(["store_id", "product_id"], sort=False):
        future = future_groups.get(keys)
        if future is None or len(future) < prediction_length:
            dropped_future += 1
            continue

        target = pd.to_numeric(context[target_col], errors="coerce").to_numpy(dtype=np.float32)
        if context_length:
            context = context.iloc[-context_length:]
            target = target[-context_length:]
        if len(target) == 0:
            dropped_short += 1
            continue
        if np.all(np.isnan(target)):
            dropped_all_nan += 1
            continue

        chronos_inputs.append(
            {
                "target": target,
                "past_covariates": pack_covariates(context),
                "future_covariates": pack_covariates(future.iloc[:prediction_length]),
            }
        )
        index_rows.append({"store_id": keys[0], "product_id": keys[1]})

        if limit_series and len(chronos_inputs) >= limit_series:
            break

    index_df = pd.DataFrame(index_rows)
    stats = {
        "target_col": target_col,
        "cutoff_date": cutoff_date,
        "prediction_length": prediction_length,
        "context_length": context_length,
        "num_series": len(chronos_inputs),
        "covariates": COVARIATE_COLS,
        "dropped_short": dropped_short,
        "dropped_all_nan": dropped_all_nan,
        "dropped_future": dropped_future,
    }
    return chronos_inputs, index_df, eval_df, stats


def load_pipeline(model_id, device_map, local_files_only, lora_dir=None):
    from chronos import BaseChronosPipeline

    pipeline = BaseChronosPipeline.from_pretrained(
        model_id,
        device_map=device_map,
        local_files_only=local_files_only,
    )
    if lora_dir:
        from peft import PeftModel

        pipeline.model = PeftModel.from_pretrained(pipeline.model, lora_dir)
        print("loaded LoRA adapter:", lora_dir)
    return pipeline


def predict_median(pipeline, chronos_inputs, prediction_length, batch_size, context_length):
    quantiles, _ = pipeline.predict_quantiles(
        chronos_inputs,
        prediction_length=prediction_length,
        quantile_levels=[0.5],
        batch_size=batch_size,
        context_length=context_length,
        cross_learning=False,
    )
    preds = np.stack([q[0, :, 0].detach().cpu().numpy() for q in quantiles]).astype(np.float32)
    np.clip(preds, 0, None, out=preds)
    return preds


def format_predictions(preds, index_df, cutoff_date, prediction_length):
    dates = pd.date_range(cutoff_date, periods=prediction_length).strftime("%Y-%m-%d")
    wide = pd.concat(
        [index_df.reset_index(drop=True), pd.DataFrame(preds, columns=dates)],
        axis=1,
    )
    return wide.melt(
        id_vars=["store_id", "product_id"],
        value_vars=list(dates),
        var_name="dt",
        value_name="prediction",
    )


def add_psd(eval_df, train_df, cutoff_date, context_length, prediction_length):
    eval_df = eval_df.copy()
    train_df = train_df.copy()
    train_df["dt"] = pd.to_datetime(train_df["dt"])
    cutoff = pd.to_datetime(cutoff_date)
    start_date = cutoff - pd.Timedelta(days=context_length + prediction_length)

    psd = (
        train_df[train_df["dt"] >= start_date]
        .groupby(["store_id", "product_id"], sort=False)["sale_amount"]
        .mean()
        .rename("psd")
        .reset_index()
    )
    return eval_df.merge(psd, on=["store_id", "product_id"])


def evaluate(pred_df, eval_df, train_df, cutoff_date, context_length, prediction_length):
    pred_df = pred_df.copy()
    eval_df = add_psd(eval_df, train_df, cutoff_date, context_length, prediction_length)
    pred_df["dt"] = pred_df["dt"].astype(str)
    eval_df["dt"] = eval_df["dt"].astype(str)

    df = eval_df[["store_id", "product_id", "dt", "sale_amount", "stock_hour6_22_cnt", "psd"]].merge(
        pred_df, on=["store_id", "product_id", "dt"]
    )
    df_eval = df.query("psd >= 0 and stock_hour6_22_cnt == 0")
    err = df_eval["prediction"] - df_eval["sale_amount"]
    metrics = {
        "sample_cnt": int(len(df_eval)),
        "wape": float(err.abs().sum() / df_eval["sale_amount"].abs().sum()),
        "wpe": float(err.sum() / df_eval["sale_amount"].abs().sum()),
        "mae": float(err.abs().mean()),
        "mse": float((err**2).mean()),
    }
    return df, pd.DataFrame([metrics])


def parse_args():
    parser = argparse.ArgumentParser(description="Chronos2 zero-shot on FreshRetailNet-LT")
    parser.add_argument("--demand_path", type=str, default=None)
    parser.add_argument("--model_id", type=str, default="amazon/chronos-2")
    parser.add_argument("--output_dir", type=str, default="./frn_results")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--cutoff_date", type=str, default="2025-07-14")
    parser.add_argument("--prediction_length", type=int, default=7)
    parser.add_argument("--context_length", type=int, default=35)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device_map", type=str, default="cuda")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--lora_dir", type=str, default=None)
    parser.add_argument("--limit_series", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--target_only", action="store_true", help="disable covariates and use target-only inputs")
    return parser.parse_args()


def main():
    args = parse_args()
    train_df, eval_df, raw_train_df, target_col = load_frn_data(args.demand_path)
    use_covariates = not args.target_only
    build_inputs = build_covariate_inputs if use_covariates else build_target_only_inputs
    chronos_inputs, index_df, eval_df, stats = build_inputs(
            train_df=train_df,
            eval_df=eval_df,
            target_col=target_col,
            cutoff_date=args.cutoff_date,
            prediction_length=args.prediction_length,
            context_length=args.context_length,
            limit_series=args.limit_series,
        )
    print("input stats:", stats)
    print("first input:", chronos_inputs[0] if chronos_inputs else None)
    # print("index head:")
    # print(index_df.head())

    if args.dry_run:
        return

    pipeline = load_pipeline(args.model_id, args.device_map, args.local_files_only, args.lora_dir)
    preds = predict_median(
        pipeline=pipeline,
        chronos_inputs=chronos_inputs,
        prediction_length=args.prediction_length,
        batch_size=args.batch_size,
        context_length=args.context_length,
    )
    pred_df = format_predictions(preds, index_df, args.cutoff_date, args.prediction_length)
    merged_df, metrics_df = evaluate(
        pred_df=pred_df,
        eval_df=eval_df,
        train_df=raw_train_df,
        cutoff_date=args.cutoff_date,
        context_length=args.context_length,
        prediction_length=args.prediction_length,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    default_name = "recovered" if args.demand_path else "censored"
    run_name = args.run_name or (f"{default_name}_covariates" if use_covariates else default_name)
    pred_path = output_dir / f"{run_name}_predictions.parquet"
    metrics_path = output_dir / f"{run_name}_metrics.parquet"
    merged_df.to_parquet(pred_path)
    metrics_df.to_parquet(metrics_path)
    print(metrics_df)
    print("saved:", pred_path)
    print("saved:", metrics_path)


if __name__ == "__main__":
    main()
