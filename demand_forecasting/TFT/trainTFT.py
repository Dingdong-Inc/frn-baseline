import os
import time, random
import numpy as np
import pandas as pd
from model import Model
import config
from dataset import Dataset
from datetime import datetime
import torch
print("get_float32_matmul_precision: ", torch.get_float32_matmul_precision())
torch.set_float32_matmul_precision('highest')
from datasets import load_dataset
from lightning.pytorch.callbacks import ModelCheckpoint
import argparse

root_path = os.path.dirname(os.path.abspath("./") + "/")
print(root_path)

# For multiple training, a new checkpoint callback must be set each time
def get_checkpoint_callback(data_type='censored', ckp_dir='checkpoints'):
    return ModelCheckpoint(
        dirpath=f"./lightning_logs/{data_type}/{ckp_dir}",
        filename="{epoch}-{step}",  # Filename format
        save_top_k=-1,  # Save all checkpoints (do not delete old files)
        every_n_epochs=1,  # Save once every epoch (optional, default is 1)
)

def run(df, config):
    fixd_seed = 2026
    random.seed(fixd_seed)
    torch.manual_seed(fixd_seed)
    np.random.seed(fixd_seed)

    t = time.time()
    dataset = Dataset(df, config)
    t0 = time.time()
    print(f"dataset generation cost {t0-t}")
    model = Model(dataset, config)
    t1 = time.time()
    print(f"model init cost {t1-t0}")
    model.train()
    t2 = time.time()
    print(f"train cost {t2-t1}")
    if config.valid:
        model.valid()
        t3 = time.time()
        print(f"validation cost {t3-t2}")

config.date = '2025-07-14'
config.use_gpu = True
config.trainer_config["default_root_dir"] = root_path


if __name__ == '__main__':
    print("Initial Start:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--demand_path",
        type=str,
        default='../../latent_demand_recovery/exp/demand/demand.parquet',
        help="demand data path, default '../../latent_demand_recovery/exp/demand/demand.parquet'"
    )
    parser.add_argument("--demand", action='store_true', help="use recoverd demand or not")
    parser.add_argument("--ckp_dir", type=str, default='checkpoints', help="checkpoints dir")
    args = parser.parse_args()

    t1 = time.time()
    if args.demand:
        config.trainer_config['callbacks'] = get_checkpoint_callback('recovered', args.ckp_dir)
        df = pd.read_parquet(args.demand_path)
        df['sale_amount'] = df['sale_amount_pred']
    else:
        config.trainer_config['callbacks'] = get_checkpoint_callback('censored', args.ckp_dir)
        dataset = load_dataset("Dingdong-Inc/FreshRetailNet-LT")
        df = dataset['train'].to_pandas()
    df = df.sort_values(['city_id', 'store_id', 'management_group_id', 'first_category_id', 'second_category_id', 'third_category_id', 'product_id', 'dt'])
    df['day_of_week'] = df['dt'].apply(lambda x : datetime.strptime(x, '%Y-%m-%d').weekday())
    print(len(df))
    t2 = time.time()
    print(f"load data cost {t2 - t1}s")

    run(df, config)

    print("Final End:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))