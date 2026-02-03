import numpy as np
import pandas as pd
import os
from datasets import load_dataset
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler



class Dataset_Custom(Dataset):
    def __init__(self, flag='train', size=None, total_seq_len=None,
                 features='MS', data_path=None,
                 target='sale_amount', scale=True, train_only=False):
        # size [seq_len, label_len, pred_len]
        print(flag, size, total_seq_len, features, data_path, target, scale, train_only)
        if size == None:
            self.seq_len = 35
            self.pred_len = 7
        else:
            self.seq_len = size[0]
            self.pred_len = size[1]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.train_only = train_only

        self.data_path = data_path
        self.interval_list = []
        self.n_window_list = []
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        if self.data_path==None:
            dataset = load_dataset("Dingdong-Inc/FreshRetailNet-LT")
            df = dataset['train'].to_pandas()
        else:
            # also support parquet file(load by pandas.read_parquet)
            df = pd.read_parquet(self.data_path)
        if 'date' not in df.columns:
            df = df.rename(columns={'dt': 'date'})
        df = df.sort_values(by=['store_id', 'product_id', 'date'])
        self.group_interval = df.groupby(['store_id', 'product_id'], sort=False)['date'].count().tolist()
        df = df[
            ['date', 'discount', 'holiday_flag', 'precpt', 'avg_temperature', 'avg_humidity',
             'avg_wind_level', self.target]] # activity_flag',
        '''
        df.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df.columns)
        if self.features == 'S':
            cols.remove(self.target)
        cols.remove('date')
        # print(cols)

        if self.features == 'M' or self.features == 'MS':
            # date + features + target
            df = df[['date'] + cols]
            # features + target
            cols_data = df.columns[1:]
            df_data = df[cols_data]
        elif self.features == 'S':
            # date + features + target
            df = df[['date'] + cols + [self.target]]
            # target
            df_data = df[[self.target]]

        train_data = []
        data_split = []
        idx = 0
        for total_seq_len in self.group_interval:
            num_train = int(total_seq_len * (0.7 if not self.train_only else 1))
            num_test = int(total_seq_len * 0.2)
            num_vali = total_seq_len - num_train - num_test
            if num_train < self.seq_len + self.pred_len:
                idx += total_seq_len
                continue

            border1s = [0, num_train - self.seq_len, total_seq_len - num_test - self.seq_len]
            border2s = [num_train, num_train + num_vali, total_seq_len]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]
            interval = border2 - border1
            n_window = border2 - border1 - self.seq_len - self.pred_len + 1
            if n_window >= 1:
                self.interval_list.append(interval if len(self.interval_list) == 0 else self.interval_list[-1] + interval)
                self.n_window_list.append(n_window if len(self.n_window_list) == 0 else self.n_window_list[-1] + n_window)

                unit = df_data.iloc[idx:idx + total_seq_len]
                train_data.append(unit.iloc[border1s[0]:border2s[0]])
                data_split.append(unit.iloc[border1:border2])
            idx += total_seq_len

        data_split = pd.concat(data_split, axis=0, ignore_index=True)
        if self.scale:
            train_data = pd.concat(train_data, axis=0, ignore_index=True)
            self.scaler.fit(train_data.values)
            data_split = self.scaler.transform(data_split.values)
            print(df.shape, df_data.shape, train_data.shape, data_split.shape)
        else:
            data_split = data_split.values

        self.data_x = data_split
        self.data_y = data_split

    def __getitem__(self, index):
        # 二分查找，线性查找会慢近20倍，组数越多越慢
        index_l, index_r = 0, len(self.n_window_list) - 1
        while index_l < index_r:
            mid = (index_l + index_r) // 2
            if index >= self.n_window_list[mid]:
                index_l = mid + 1
            else:
                index_r = mid
        dataset_index = index_l

        index = index - self.n_window_list[dataset_index - 1] if dataset_index > 0 else index
        his_interval = self.interval_list[dataset_index - 1] if dataset_index > 0 else 0

        s_begin = his_interval + index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        return seq_x, seq_y

    def __len__(self):
        return self.n_window_list[-1]

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Pred(Dataset):
    def __init__(self, flag='pred', size=None, total_seq_len=None,
                 features='MS', data_path='train.parquet',
                 target='sale_amount', scale=True,
                 inverse=False,
                 cols=None, train_only=False):
        # size [seq_len, label_len, pred_len]
        if size == None:
            self.seq_len = 35
            self.pred_len = 7
        else:
            self.seq_len = size[0]
            self.pred_len = size[1]
        # init
        assert flag in ['pred']

        self.features = features  # 'MS'
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.cols = cols
        self.data_path = data_path
        self.interval_list = []
        self.n_window_list = []
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()

        dataset = load_dataset("Dingdong-Inc/FreshRetailNet-LT")
        df = dataset['train'].to_pandas()
        df_eval = dataset['eval'].to_pandas()
        if self.data_path != None:
            df = pd.read_parquet(self.data_path)
        df_eval_group_ids = df_eval[['store_id', 'product_id']].drop_duplicates()
        df = pd.merge(df, df_eval_group_ids, on = ['store_id', 'product_id'])
        if 'date' not in df.columns:
            df = df.rename(columns={'dt': 'date'})
        df = df.sort_values(by=['store_id', 'product_id', 'date'])
        self.group_interval = df.groupby(['store_id', 'product_id'], sort=False)['date'].count().tolist()
        df = df[
            ['date', 'discount', 'holiday_flag', 'precpt', 'avg_temperature', 'avg_humidity',
             'avg_wind_level', self.target]] # 'activity_flag'
        '''
        df.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
        else:
            cols = list(df.columns)
            self.cols = cols.copy()
            cols.remove('date')
        if self.features == 'S':
            cols.remove(self.target)

        if self.features == 'M' or self.features == 'MS':
            df = df[['date'] + cols]
            cols_data = df.columns[1:]
            df_data = df[cols_data]
        elif self.features == 'S':
            df = df[['date'] + cols + [self.target]]
            df_data = df[[self.target]]

        data_split = []
        idx = 0
        for total_seq_len in self.group_interval:
            border1 = total_seq_len - self.seq_len
            border2 = total_seq_len
            interval = border2 - border1
            n_window = 1
            if border1 >= 0:
                self.interval_list.append(interval if len(self.interval_list) == 0 else self.interval_list[-1] + interval)
                self.n_window_list.append(n_window if len(self.n_window_list) == 0 else self.n_window_list[-1] + n_window)
                unit = df_data.iloc[idx:idx + total_seq_len]
                data_split.append(unit.iloc[border1:border2])
            idx += total_seq_len

        data_split = pd.concat(data_split, axis=0, ignore_index=True)
        if self.scale:
            self.scaler.fit(df_data.values)
            data_split = self.scaler.transform(data_split.values)
        else:
            data_split = data_split.values

        self.data_x = data_split
        self.data_y = data_split

    def __getitem__(self, index):
        # 二分查找
        index_l, index_r = 0, len(self.n_window_list) - 1
        while index_l < index_r:
            mid = (index_l + index_r) // 2
            if index >= self.n_window_list[mid]:
                index_l = mid + 1
            else:
                index_r = mid
        dataset_index = index_l

        index = index - self.n_window_list[dataset_index - 1] if dataset_index > 0 else index
        his_interval = self.interval_list[dataset_index - 1] if dataset_index > 0 else 0

        s_begin = his_interval + index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_begin]
        return seq_x, seq_y

    def __len__(self):
        return self.n_window_list[-1]

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
