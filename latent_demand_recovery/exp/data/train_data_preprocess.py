import os
import pandas as pd
import numpy as np
from datasets import load_dataset


def preprocess():
    dataset = load_dataset("Dingdong-Inc/FreshRetailNet-LT")
    data = dataset['train'].to_pandas()
    data = data.sort_values(by=['store_id', 'product_id', 'dt'])
    data['date'] = pd.to_datetime(data['dt'])
    data['idx'] = data.groupby(['store_id', 'product_id']).cumcount()
    data['min_date'] = data.groupby(['store_id', 'product_id'])['date'].transform('min')
    data['dt_idx'] = (data['date'] - data['min_date']).dt.days
    data['idx_gap'] = data['dt_idx'] - data['idx']
    data['sequence_id'] = data.groupby(['store_id', 'product_id', 'idx_gap']).ngroup()
    data['sequence_length'] = data.groupby('sequence_id')['dt'].transform('count')
    long_data = data[data['sequence_length']>=30]
    short_data = data[data['sequence_length']<30]

    long_data['sub_idx'] = long_data.groupby('sequence_id').cumcount()
    long_data_0 = long_data.query('sub_idx<30')
    long_data_0['sub_sequence_id'] = 0
    long_data_1 = long_data[long_data['sub_idx'] >= long_data['sequence_length'] % 30]
    long_data_1['sub_idx'] = long_data_1.groupby('sequence_id').cumcount()
    long_data_1['sub_sequence_id'] = long_data_1['sub_idx']//30 + 1
    long_data_slice = pd.concat([long_data_0, long_data_1], axis=0)

    city_df = data.groupby(['city_id', 'date', 'dt', 'holiday_flag']).agg({'precpt':'mean', 'avg_temperature':'mean', 'avg_wind_level':'mean', 'avg_humidity':'mean'}).reset_index()
    city_df.columns = ['city_id', 'date', 'dt', 'holiday_flag', 'precpt_city', 'avg_temperature_city', 'avg_wind_level_city', 'avg_humidity_city']

    left_df = city_df.merge(short_data[['sequence_id', 'store_id', 'product_id', 'city_id', 'min_date']].drop_duplicates(), on='city_id')
    left_df['day_diff'] = (left_df['date'] - left_df['min_date']).dt.days
    left_df = left_df[(left_df['day_diff']<30)&(left_df['day_diff']>=0)]

    short_data_slice = left_df.merge(short_data[['sequence_id', 'date', 'hours_sale', 'hours_stock_status', 'precpt', 'avg_temperature']], on=['sequence_id', 'date'], how='left')

    short_data_slice['precpt'] = short_data_slice['precpt'].fillna(short_data_slice['precpt_city'])
    short_data_slice['avg_temperature'] = short_data_slice['avg_temperature'].fillna(short_data_slice['avg_temperature_city'])

    def fill_array(x, fillna=0.0):
        # 定义目标填充值：24个0的数组
        fill_val = [fillna]*24
        # 情况1：空值（None/np.nan/空列表）→ 返回填充值
        # print(x, isinstance(x, list))
        if not isinstance(x, np.ndarray):
            return fill_val
        return x


    short_data_slice['hours_sale'] = short_data_slice['hours_sale'].apply(lambda x:fill_array(x, 0.0)).tolist()
    short_data_slice['hours_stock_status'] = short_data_slice['hours_stock_status'].apply(lambda x:fill_array(x, 1.0)).tolist()
    short_data_slice = short_data_slice.sort_values(by=['sequence_id', 'date'])
    short_data_slice['sub_sequence_id'] = 0
    short_data_slice['discount'] = 1.0

    effective_cols = ['city_id', 'date', 'holiday_flag', 'sequence_id', 'sub_sequence_id', 'store_id',
           'product_id', 'hours_sale', 'discount', 
           'hours_stock_status', 'precpt', 'avg_temperature']

    data_slice = pd.concat([long_data_slice[effective_cols], short_data_slice[effective_cols]], axis=0)
    data_slice.to_parquet('./data_slice.parquet')
    return data_slice