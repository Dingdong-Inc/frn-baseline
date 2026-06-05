#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

cd "${ROOT_DIR}"

echo "[1/5] Chronos2 zero-shot: censored + covariates"
conda run -n py311 python demand_forecasting/chronos2/frn_zero_shot.py \
  --local_files_only \
  --run_name censored_covariates

echo "[2/5] Chronos2 zero-shot: DLinear recovered + covariates"
conda run -n py311 python demand_forecasting/chronos2/frn_zero_shot.py \
  --local_files_only \
  --demand_path latent_demand_recovery/exp/demand/DLinear_demand.parquet \
  --run_name DLinear_covariates

echo "[3/5] Chronos2 zero-shot: iTransformer recovered + covariates"
conda run -n py311 python demand_forecasting/chronos2/frn_zero_shot.py \
  --local_files_only \
  --demand_path latent_demand_recovery/exp/demand/iTransformer_demand.parquet \
  --run_name iTransformer_covariates

echo "[4/5] Chronos2 zero-shot: ImputeFormer recovered + covariates"
conda run -n py311 python demand_forecasting/chronos2/frn_zero_shot.py \
  --local_files_only \
  --demand_path latent_demand_recovery/exp/demand/ImputeFormer_demand.parquet \
  --run_name ImputeFormer_covariates

echo "[5/5] Summarize Chronos2 metrics"
python - <<'PY'
from pathlib import Path
import pandas as pd

rows = []
for p in sorted(Path("frn_results").glob("*_metrics.parquet")):
    if p.name == "chronos2_summary_metrics.parquet":
        continue
    method = p.name.replace("_metrics.parquet", "")
    df = pd.read_parquet(p).copy()
    df.insert(0, "method", method)
    rows.append(df)

summary = pd.concat(rows, ignore_index=True).sort_values("method").reset_index(drop=True)
summary.to_csv("frn_results/chronos2_summary_metrics.csv", index=False)
summary.to_parquet("frn_results/chronos2_summary_metrics.parquet", index=False)
print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
PY

echo "Done. Summary saved to:"
echo "  frn_results/chronos2_summary_metrics.csv"
echo "  frn_results/chronos2_summary_metrics.parquet"
