import pandas as pd
import numpy as np
from datasets import load_dataset
import argparse

def ssa_predict(demand_df, target_col='sale_amount'):
    dataset = load_dataset("Dingdong-Inc/FreshRetailNet-LT")
    eval_df = dataset['eval'].to_pandas()
    demand_df = demand_df.merge(eval_df[['store_id', 'product_id']].drop_duplicates(), on=['store_id', 'product_id'])
    all_data = pd.concat([demand_df, eval_df], axis=0)
    all_data = all_data.sort_values(by=['store_id', 'product_id', 'dt'])
    all_data = all_data.query('dt>="2025-06-02"')

    # calendar feature
    all_data['time_idx'] = (pd.to_datetime(all_data['dt']) - pd.to_datetime(all_data['dt']).min()).dt.days
    all_data['dow'] = pd.to_datetime(all_data['dt']).dt.dayofweek
    # all_data.loc[all_data['dt'].isin(['2024-05-02','2024-05-03','2024-05-04']), 'holiday_flag']=0
    all_data.loc[all_data['holiday_flag']==1, 'dow'] = 5
    all_data.loc[(all_data['holiday_flag']==0)&(all_data['dow'].isin([5,6])), 'dow'] = 0
    # precipitation feature
    all_data['is_precpt'] = (all_data['precpt']>3.5).astype(int)
    # continuouse and catigorical variables
    x_cont = all_data[['time_idx', 'discount', target_col]].values.reshape(-1, 49, 3)
    x_cat = all_data[['dow', 'holiday_flag', 'is_precpt']].values.reshape(-1, 49, 3)
    # config
    max_encoder_length=42
    time_idx=0
    discount_idx=1
    dow_idx=0
    holiday_idx=1
    is_precpt_idx=2
    holiday_label = 5
    # weight
    date_distance = 10 - x_cont[:, :max_encoder_length, time_idx][:,None,:]/4
    discount_distance = 1 - np.abs((x_cont[:, max_encoder_length:, discount_idx][...,None] - x_cont[:, :max_encoder_length, discount_idx][:,None,:]))
    decoder_dow = (x_cat[:, max_encoder_length:, dow_idx][...,None])
    encoder_dow = (x_cat[:, :max_encoder_length, dow_idx][:,None,:])
    dow_distance = decoder_dow - encoder_dow
    decoder_precpt = (x_cat[:, max_encoder_length:, is_precpt_idx][...,None])
    encoder_precpt = (x_cat[:, :max_encoder_length, is_precpt_idx][:,None,:])
    precpt_distance = decoder_precpt - encoder_precpt
    date_distance = np.where((decoder_dow == holiday_label).astype(int) - (encoder_dow == holiday_label).astype(int) == 0, date_distance - 5, date_distance)
    date_distance = np.where((decoder_dow != holiday_label) & (encoder_dow == holiday_label), date_distance + 5, date_distance)
    date_distance = np.where((decoder_dow == holiday_label) & (encoder_dow != holiday_label), date_distance + 5, date_distance)
    date_distance = np.where(dow_distance == 0, date_distance - 3, date_distance)
    date_distance = np.where((precpt_distance == 0) & (decoder_precpt == 1), date_distance - 3, date_distance)
    date_distance = np.where(precpt_distance != 0, date_distance + 3, date_distance)
    date_distance = np.clip(date_distance, a_min=1, a_max=10)
    weight = (10-date_distance) * discount_distance**2
    weight = np.exp(weight)/np.exp(weight).sum(axis=-1, keepdims=True)

    # high sale & low sale info
    mu = all_data.query("dt<='2025-07-13'").groupby(['store_id', 'product_id']).agg({'sale_amount':'mean'})
    mu = mu.reset_index().rename(columns={'sale_amount':'psd'})
    all_data = all_data.merge(mu, on=['store_id', 'product_id'], how='left')

    pdf = all_data.query("dt>='2025-07-14'").copy()
    target_idx = 2
    target = x_cont[:, :max_encoder_length, target_idx:target_idx+1]
    index = encoder_dow.squeeze()[...,None] == np.arange(6)
    mean_sale = np.nanmean(np.where(index, target, np.nan), axis=1) + 0.001
    season_ratio = np.nanmedian(mean_sale[...,None] / mean_sale[:,None,...], axis=0)
    ratio = season_ratio[decoder_dow, encoder_dow]
    pred = (weight * ratio)@target
    # import ipdb; ipdb.set_trace()
    pdf[f'ssa_pred_{target_col}'] = pred.reshape(-1)
    # overall
    metric = evaluation(pdf, target_col), # overall
    print(metric)


def evaluation(pdf, target_col='sale_amount'):
    res = []
    pred_col = f'ssa_pred_{target_col}'
    wape_list,mae_list,wpe_list = [],[],[]
    df = pdf.query('stock_hour6_22_cnt==0')
    mae = (df['sale_amount'] - df[pred_col]).abs().sum()
    wape = mae / df['sale_amount'].sum()
    mae = (df['sale_amount'] - df[pred_col]).abs().mean()
    mse = np.power(df['sale_amount'] - df[pred_col], 2).mean()
    wpe = (df[pred_col]-df['sale_amount']).sum()/df['sale_amount'].sum()
    metric = {'wape':wape, 'wpe':wpe, 'mae':mae, 'mse':mse}
    return metric

def load_data(path):
    if path:
        dataset = load_dataset("Dingdong-Inc/FreshRetailNet-50K")
        data = dataset['train'].to_pandas()
    return data
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--demand_path", 
        type=str, 
        default='../../latent_demand_recovery/exp/demand/DLinear_demand.parquet',
        help="demand data path, default '../../latent_demand_recovery/exp/demand/DLinear_demand.parquet'"
    )
    parser.add_argument(
        "--demand",
        action='store_true',
        help="use recoverd demand or not"
    )
    args = parser.parse_args()
    if args.demand:
        target_col = 'sale_amount_pred'
        demand_df = pd.read_parquet(args.demand_path)
    else:
        target_col = 'sale_amount'
        dataset = load_dataset("Dingdong-Inc/FreshRetailNet-LT")
        demand_df = dataset['train'].to_pandas()
    ssa_predict(demand_df, target_col)