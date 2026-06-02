from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional
from tqdm.auto import tqdm

import numpy as np
import pandas as pd

import math

from kernel_synth_local import generate_time_series

# Global cache for the price-volume table.
_PRICE_VOLUME_TABLE = None

def load_price_volume_table(csv_path: str = "price_volume_table.csv"):
    """Load the price-volume table and parse price ranges plus logit parameters."""
    global _PRICE_VOLUME_TABLE
    if _PRICE_VOLUME_TABLE is not None:
        return _PRICE_VOLUME_TABLE

    df = pd.read_csv(csv_path)
    rows = []
    for _, row in df.iterrows():
        price_range = row["price_type"]
        low, high = price_range.split("~")
        low, high = float(low), float(high)

        logit_str = row["logit_params_w"]
        a, b = map(float, logit_str.split(","))

        rows.append({
            "group": row["purchase_management_group_name"],
            "price_min": low,
            "price_max": high,
            "a": a,
            "b": b,
        })

    _PRICE_VOLUME_TABLE = pd.DataFrame(rows)
    return _PRICE_VOLUME_TABLE


def get_logit_params(group_name: str, price: float, table: pd.DataFrame):
    """Find logit parameters a,b by group name and price."""
    mask = (table["group"] == group_name) & (table["price_min"] <= price) & (price < table["price_max"])
    matched = table[mask]
    if len(matched) == 0:
        # Fallback: use the last interval for this group, which has the largest price range.
        matched = table[table["group"] == group_name].iloc[-1:]
    return matched.iloc[0]["a"], matched.iloc[0]["b"]


def generate_group_and_price(rng: np.random.Generator, table: pd.DataFrame):
    """Randomly sample a purchase management group and price."""
    groups = table["group"].unique()
    group_name = rng.choice(groups)

    # Get all price intervals for the sampled group.
    group_rows = table[table["group"] == group_name]

    # Sample one interval, then sample a price inside that interval.
    row = group_rows.sample(1, random_state=rng.integers(0, 2**31)).iloc[0]
    price = float(rng.uniform(row["price_min"], row["price_max"]))

    # Cap prices at 120.
    price = min(price, 120.0)

    return group_name, price


@dataclass
class CovariateTaskRecord:
    task_id: str
    start: str
    freq: str
    prediction_length: int

    target_name: str
    target: List[float]

    # Known future covariates.
    known_future_covariates: Dict[str, List[float]]

    static_categorical: Dict[str, str]
    static_real: Dict[str, float]

    def to_dict(self) -> Dict:
        return asdict(self)


def minmax_scale(x: np.ndarray, low: float = 0.0, high: float = 1.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    xmin, xmax = x.min(), x.max()
    if float(xmax - xmin) < 1e-8:
        return np.full_like(x, fill_value=(low + high) / 2, dtype=np.float32)
    z = (x - xmin) / (xmax - xmin)
    return (low + (high - low) * z).astype(np.float32)


def resize_series(x: np.ndarray, length: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if len(x) == length:
        return x
    if len(x) > length:
        return x[:length]

    pad = np.full(length - len(x), x[-1], dtype=np.float32)
    return np.concatenate([x, pad]).astype(np.float32)


# =========================
# Base generators
# =========================

def generate_kernel_synth_series(length: int, max_kernels: int = 5) -> np.ndarray:
    sample = generate_time_series(max_kernels=max_kernels)
    target = np.asarray(sample["target"], dtype=np.float32)
    return resize_series(target, length)


def generate_ar_series(
    length: int,
    phi: float = 0.8,
    sigma: float = 0.5,
    intercept: float = 0.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.zeros(length, dtype=np.float32)
    noise = rng.normal(0.0, sigma, size=length)
    x[0] = intercept + noise[0]
    for t in range(1, length):
        x[t] = intercept + phi * x[t - 1] + noise[t]
    return x.astype(np.float32)


def generate_simple_ets_series(
    length: int,
    level: float = 5.0,
    trend: float = 0.02,
    season_period: int = 7,
    season_amp: float = 1.5,
    noise_std: float = 0.25,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Simplified ETS style:
    level + trend + additive seasonality + noise
    """
    rng = np.random.default_rng(seed)
    t = np.arange(length)

    base_level = level + trend * t
    seasonality = season_amp * np.sin(2 * np.pi * t / season_period)
    noise = rng.normal(0.0, noise_std, size=length)

    y = base_level + seasonality + noise
    return y.astype(np.float32)


def generate_simple_tsi_series(
    length: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Simplified TSI:
    Trend + Seasonality + Irregularity
    """
    rng = np.random.default_rng(seed)
    t = np.arange(length)

    # Trend
    trend_type = rng.choice(["linear_up", "linear_down", "flat", "piecewise"])
    if trend_type == "linear_up":
        trend = rng.uniform(0.005, 0.03) * t
    elif trend_type == "linear_down":
        trend = -rng.uniform(0.003, 0.02) * t
    elif trend_type == "flat":
        trend = np.zeros(length)
    else:
        cp = int(rng.integers(length // 4, 3 * length // 4))
        slope1 = rng.uniform(-0.01, 0.02)
        slope2 = rng.uniform(-0.01, 0.03)
        trend = np.concatenate([
            slope1 * np.arange(cp),
            slope1 * cp + slope2 * np.arange(length - cp),
        ])

    # Seasonality
    season = np.zeros(length)
    season += rng.uniform(0.5, 2.0) * np.sin(2 * np.pi * t / 7.0 + rng.uniform(0, np.pi))
    if rng.random() < 0.3:
        season += rng.uniform(0.2, 1.2) * np.sin(2 * np.pi * t / 30.0 + rng.uniform(0, np.pi))

    # Irregularity
    noise = rng.normal(0.0, rng.uniform(0.15, 0.6), size=length)

    # Spike / outlier
    spikes = np.zeros(length)
    if rng.random() < 0.7:
        num_spikes = int(rng.integers(1, max(2, length // 40)))
        spike_idx = rng.choice(np.arange(length), size=num_spikes, replace=False)
        spikes[spike_idx] = rng.normal(0.0, 2.5, size=num_spikes)

    y = trend + season + noise + spikes
    return y.astype(np.float32)


def generate_base_univariate(
    length: int,
    max_kernels: int = 5,
    generator_type: Optional[str] = None,
    seed: Optional[int] = None,
) -> tuple[np.ndarray, str]:
    """
    Pick one of four generators:
    - kernelsynth
    - ar
    - ets
    - tsi

    Returns:
    - series
    - generator_name
    """
    rng = np.random.default_rng(seed)

    if generator_type is None:
        generator_type = rng.choice(
            ["kernelsynth", "ar", "ets", "tsi"],
            p=[0.35, 0.20, 0.20, 0.25],
        )

    if generator_type == "kernelsynth":
        return generate_kernel_synth_series(length=length, max_kernels=max_kernels), "kernelsynth"

    if generator_type == "ar":
        return generate_ar_series(
            length=length,
            phi=float(rng.uniform(0.7, 0.99)),
            sigma=float(rng.uniform(0.15, 0.8)),
            intercept=float(rng.uniform(-1.0, 1.0)),
            seed=seed,
        ), "ar"

    if generator_type == "ets":
        return generate_simple_ets_series(
            length=length,
            level=float(rng.uniform(2.0, 8.0)),
            trend=float(rng.uniform(-0.01, 0.04)),
            season_period=int(rng.choice([7])), # Keep only 7-day seasonality.
            season_amp=float(rng.uniform(0.3, 2.0)),
            noise_std=float(rng.uniform(0.1, 0.5)),
            seed=seed,
        ), "ets"

    if generator_type == "tsi":
        return generate_simple_tsi_series(length=length, seed=seed), "tsi"

    raise ValueError(f"Unsupported generator_type: {generator_type}")


# =========================
# Covariate task builder
# =========================

def build_cotemporaneous_covariate_task(
    base_signal: np.ndarray,
    task_id: str,
    prediction_length: int,
    generator_name: str,
    start: str = "2020-01-01",
    freq: str = "D",
    seed: Optional[int] = None,
    covariate_mode: str = "both",  # "discount_only", "rainfall_only", "both"
) -> CovariateTaskRecord:
    """
    Keep only cotemporaneous effects:
    discount / rainfall / holiday directly affect sales at the same timestamp.

    static features:
    - purchase_management_group_name: purchase management group
    - price: static real value sampled from the price-volume table, capped at 120
    - base_generator: static categorical label

    known future covariates:
    - discount: 2-5 day block events, value 0-1 (0.8 means 20% off)
    - rainfall: 1-3 day block events, value 0-20, mostly 0-5
    - holiday: first 2 days of every 7-day period are set to 1
    """
    rng = np.random.default_rng(seed)
    n = len(base_signal)
    t = np.arange(n)

    # =========================================================
    # 1) Load the price-volume table.
    # =========================================================
    pv_table = load_price_volume_table()

    # =========================================================
    # 2) static features
    # =========================================================
    # Randomly sample a group and price from the price-volume table.
    group_name, price = generate_group_and_price(rng, pv_table)

    # Find the matching logit parameters.
    a, b = get_logit_params(group_name, price, pv_table)

    # =========================================================
    # 3) Known future covariates, generated according to mode.
    # =========================================================
    # holiday: first 2 days of every 7-day period are treated as holidays.
    holiday = np.zeros(n, dtype=np.float32)
    holiday[(t % 7 == 0) | (t % 7 == 1)] = 1.0

    # discount: generated according to mode.
    if covariate_mode in ("discount_only", "both"):
        discount = np.ones(n, dtype=np.float32)
        discount_start_candidates = np.arange(7, max(8, n - 7))
        num_discount_events = max(1, n // 5)  # Roughly one discount event every 5 days.

        if len(discount_start_candidates) > 0:
            discount_starts = rng.choice(
                discount_start_candidates,
                size=min(num_discount_events, len(discount_start_candidates)),
                replace=False,
            )
            for s in discount_starts:
                width = int(rng.integers(2, 6))  # 2-5 days

                # Sample discount strength from an empirical-style distribution.
                p = float(rng.uniform(0.0, 1.0))
                if p < 0.8:
                    intensity = float(rng.uniform(0.85, 0.98))  # 80% small discounts
                elif p < 0.95:
                    intensity = float(rng.uniform(0.7, 0.85))   # 15% medium discounts
                else:
                    intensity = float(rng.uniform(0.5, 0.7))     # 5% large discounts

                discount[s:min(n, s + width)] = intensity
    else:
        # rainfall_only mode: discount is always 1, meaning no discount.
        discount = np.ones(n, dtype=np.float32)

    # rainfall: generated according to mode.
    if covariate_mode in ("rainfall_only", "both"):
        rainfall = np.zeros(n, dtype=np.float32)
        rainfall_start_candidates = np.arange(3, max(4, n - 3))
        num_rain_events = max(1, n // 7)  # Roughly one rain event every 7 days.

        if len(rainfall_start_candidates) > 0:
            rainfall_starts = rng.choice(
                rainfall_start_candidates,
                size=min(num_rain_events, len(rainfall_start_candidates)),
                replace=False,
            )
            for s in rainfall_starts:
                width = int(rng.integers(1, 8))  # 1-7 days
                rain_regime = rng.choice(
                    ["light", "medium", "heavy"],
                    p=[0.82, 0.14, 0.04],
                )
                if rain_regime == "light":
                    center = float(rng.uniform(0.26, 6.5))
                elif rain_regime == "medium":
                    center = float(rng.uniform(6.5, 15.6))
                else:
                    center = float(rng.uniform(15.6, 26.0))
                for k in range(width):
                    idx = s + k
                    if idx >= n:
                        break
                    value = center + float(rng.normal(0.0, 0.12 * max(center, 1.0)))
                    rainfall[idx] = np.clip(value, 0.0, 26.0)
    else:
        # discount_only mode: rainfall is always 0.
        rainfall = np.zeros(n, dtype=np.float32)

    rainfall = rainfall.astype(np.float32)

    # =========================================================
    # 4) Target construction with logit discount effects from the price-volume table.
    # =========================================================
    # Scale the base signal into a stable range.
    base_scaled = minmax_scale(base_signal, 0.0, 10.0)

    # Base sales without discount or rainfall effects.
    base_sales = 3.0 + base_scaled

    # rainfall: positive but mild.
    rainfall_effect = 0.55 * np.sqrt(rainfall)

    # holiday: positive.
    holiday_effect = 1.2 * holiday

    # Sales before discount, but with rainfall and holiday effects.
    sales_no_discount = base_sales + rainfall_effect + holiday_effect

    # Compute discount uplift point by point.
    sales = np.zeros(n, dtype=np.float32)
    for i in range(n):
        discount_t = discount[i]
        # Logit discount effect formula.
        t = math.exp(b * (discount_t - 1))
        uplift_t = a * t / (t + a - 1)
        sales[i] = sales_no_discount[i] * uplift_t

    sales = np.maximum(0.0, sales).astype(np.float32)

    # =========================================================
    # 5) package
    # =========================================================
    static_categorical = {
        "base_generator": generator_name,
        "purchase_management_group_name": group_name,
        "covariate_mode": covariate_mode,
    }

    static_real = {
        "price": price,
    }

    return CovariateTaskRecord(
        task_id=task_id,
        start=start,
        freq=freq,
        prediction_length=prediction_length,
        target_name="sales",
        target=sales.tolist(),
        known_future_covariates={
            "discount": discount.tolist(),
            "rainfall": rainfall.tolist(),
            "holiday": holiday.tolist(),
        },
        static_categorical=static_categorical,
        static_real=static_real,
    )



def tasks_to_long_dataframe(tasks: List[CovariateTaskRecord]) -> pd.DataFrame:
    rows = []
    for task in tasks:
        n = len(task.target)
        ts = pd.date_range(task.start, periods=n, freq=task.freq)

        for i in range(n):
            row = {
                "task_id": task.task_id,
                "timestamp": ts[i],
                "target": float(task.target[i]),
            }

            for k, v in task.known_future_covariates.items():
                row[k] = float(v[i])

            row.update(task.static_categorical)
            row.update(task.static_real)
            rows.append(row)

    return pd.DataFrame(rows)


def main():
    output_dir = Path("/data/lihaoran26/chronos2/chronos2_train_data")
    output_dir.mkdir(parents=True, exist_ok=True)

    num_tasks = 1000
    context_length = 256
    prediction_length = 28
    total_length = context_length + prediction_length
    max_kernels = 5

    tasks = []

    for i in tqdm(range(num_tasks), desc="Generating tasks"):
        # Select modes in a 1:1:1 ratio.
        mode = ["discount_only", "rainfall_only", "both"][i % 3]

        # Generate the base univariate series.
        base, generator_name = generate_base_univariate(
            length=total_length,
            max_kernels=max_kernels,
            generator_type=None,   # None = randomly sample one of four generators.
            seed=2026 + i,
        )

        # Build the covariate task.
        task = build_cotemporaneous_covariate_task(
            base_signal=base,
            task_id=f"task_{i:06d}",
            prediction_length=prediction_length,
            generator_name=generator_name,
            start="2020-01-01",
            freq="D",
            seed=9000 + i,
            covariate_mode=mode,
        )
        tasks.append(task)

    jsonl_path = output_dir / "covariate_tasks.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for task in tasks:
            f.write(json.dumps(task.to_dict(), ensure_ascii=False) + "\n")

    df = tasks_to_long_dataframe(tasks)
    csv_path = output_dir / "covariate_tasks_long.csv"
    df.to_csv(csv_path, index=False)

    print(f"Saved JSONL to: {jsonl_path}")
    print(f"Saved CSV to:   {csv_path}")
    print(df.head())
    print(df["base_generator"].value_counts())


if __name__ == "__main__":
    main()
