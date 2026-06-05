#!/usr/bin/env bash
set -euo pipefail

# Build Chronos2 LoRA training data in one pass:
# 1) Download/cache official univariate data
# 2) Build the FRN management-group discount response table
# 3) Generate FRN-style synthetic data
# 4) Build real FRN retail train/validation inputs
# 5) Merge final train_inputs.pkl / val_inputs.pkl

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

PYTHON_CMD=(conda run -n py311 python)

OUTPUT_DIR="${OUTPUT_DIR:-frn_results/chronos2_lora_data}"
DEMAND_PATH="${DEMAND_PATH:-raw}"
TARGET_COL="${TARGET_COL:-sale_amount}"
OFFICIAL_MAX_PER_SOURCE="${OFFICIAL_MAX_PER_SOURCE:-5000}"
SYNTHETIC_NUM_SERIES="${SYNTHETIC_NUM_SERIES:-20000}"
HISTORY_LENGTH="${HISTORY_LENGTH:-84}"
PREDICTION_LENGTH="${PREDICTION_LENGTH:-7}"
MIN_LENGTH="${MIN_LENGTH:-91}"
TRAIN_END_DATE="${TRAIN_END_DATE:-2025-07-06}"
VAL_END_DATE="${VAL_END_DATE:-2025-07-13}"
SEED="${SEED:-2026}"
FORCE_OFFICIAL="${FORCE_OFFICIAL:-0}"

mkdir -p "$OUTPUT_DIR"

echo "[1/5] Download/cache official univariate data"
OFFICIAL_ARGS=(
  demand_forecasting/chronos2/train_data_builder/download_official_univariate.py
  --output_dir "$OUTPUT_DIR/official_univariate"
  --max_per_source "$OFFICIAL_MAX_PER_SOURCE"
)
if [[ "$FORCE_OFFICIAL" == "1" ]]; then
  OFFICIAL_ARGS+=(--force)
fi
"${PYTHON_CMD[@]}" "${OFFICIAL_ARGS[@]}"

echo "[2/5] Build FRN management-group discount response table"
"${PYTHON_CMD[@]}" demand_forecasting/chronos2/train_data_builder/frn_discount_response.py \
  --input_path "$DEMAND_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --target_col "$TARGET_COL"

echo "[3/5] Generate FRN-style synthetic data"
"${PYTHON_CMD[@]}" demand_forecasting/chronos2/train_data_builder/frn_covariate_task_generator.py \
  --response_path "$OUTPUT_DIR/frn_discount_response.parquet" \
  --output_path "$OUTPUT_DIR/synthetic_inputs.pkl" \
  --stats_path "$OUTPUT_DIR/synthetic_inputs_stats.json" \
  --num_series "$SYNTHETIC_NUM_SERIES" \
  --history_length "$HISTORY_LENGTH" \
  --prediction_length "$PREDICTION_LENGTH" \
  --seed "$SEED"

echo "[4/5] Build real FRN retail train/validation inputs"
"${PYTHON_CMD[@]}" demand_forecasting/chronos2/train_data_builder/frn_retail_input_builder.py \
  --input_path "$DEMAND_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --target_col "$TARGET_COL" \
  --train_end_date "$TRAIN_END_DATE" \
  --val_end_date "$VAL_END_DATE" \
  --prediction_length "$PREDICTION_LENGTH" \
  --min_length "$MIN_LENGTH"

echo "[5/5] Merge final LoRA training data"
"${PYTHON_CMD[@]}" demand_forecasting/chronos2/train_data_builder/frn_lora_train_data_builder.py \
  --data_dir "$OUTPUT_DIR" \
  --official_dir "$OUTPUT_DIR/official_univariate" \
  --seed "$SEED"

echo "Done. Output files:"
echo "  $OUTPUT_DIR/train_inputs.pkl"
echo "  $OUTPUT_DIR/val_inputs.pkl"
echo "  $OUTPUT_DIR/build_stats.json"
