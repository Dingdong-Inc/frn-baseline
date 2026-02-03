#!/bin/bash

# Define parameters
SEQ_LEN=35
MOVING_AVG=28
PRED_LEN=7
ENC_IN=7
MODEL_NAME="DLinear"


# Run
python -u main.py \
    --is_training 1 \
    --train_only True \
    --model "$MODEL_NAME" \
    --scale True \
    --loss 'mae' \
    --features "MS" \
    --data "recovered_dataset" \
    --data_path "../../latent_demand_recovery/exp/demand/demand.parquet" \
    --target "sale_amount_pred" \
    --revin False \
    --seq_len "$SEQ_LEN" \
    --pred_len "$PRED_LEN" \
    --moving_avg "$MOVING_AVG" \
    --enc_in "$ENC_IN" \
    --des "original_l1" \
    --do_predict \
    --itr 1 \
    --batch_size 1024 \
    --train_epochs 3 \
    --num_workers 16 \
    --learning_rate 0.001
