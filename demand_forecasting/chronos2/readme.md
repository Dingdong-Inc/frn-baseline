# Chronos2

> Chronos2 is a time series foundation model used here for FreshRetailNet demand forecasting with zero-shot prediction and LoRA fine-tuning.
> - Model: `amazon/chronos-2`

Use the existing Conda environment:

```bash
conda activate py311
export HF_ENDPOINT=https://hf-mirror.com
```

To run Chronos2 on censored/recovered sales, use:

```bash
# Perform zero-shot forecasting on censored and recovered demand inputs.
python demand_forecasting/chronos2/run_frn_predict.py

# Train Chronos2 LoRA. By default this uses raw FreshRetailNet sales
# as the FRN demand sequence, builds LoRA training data, then trains the adapter.
python demand_forecasting/chronos2/run_frn_lora_train.py

# To train LoRA on a recovered demand sequence instead, pass its parquet path.
python demand_forecasting/chronos2/run_frn_lora_train.py \
  --demand_path latent_demand_recovery/exp/demand/ImputeFormer_demand.parquet \
  --target_col sale_amount_pred

# Perform forecasting with the trained LoRA adapter.
python demand_forecasting/chronos2/run_frn_predict.py \
  --lora_dir frn_results/chronos2_lora_ckpt/frn-lora-ckpt
```

By default, prediction runs these inputs when the corresponding recovered demand files exist:

```text
censored
DLinear
iTransformer
ImputeFormer
TimesNet
```

To run one input only:

```bash
python demand_forecasting/chronos2/run_frn_predict.py --method ImputeFormer

python demand_forecasting/chronos2/run_frn_predict.py \
  --method ImputeFormer \
  --lora_dir frn_results/chronos2_lora_ckpt/frn-lora-ckpt
```
