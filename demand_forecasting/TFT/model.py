import warnings
warnings.filterwarnings("ignore")  # avoid printing out absolute paths

import torch
import numpy as np
import pandas as pd

# import for training
# import pytorch_lightning as pl
import lightning.pytorch as pl
# from lightning.pytorch.loggers import TensorBoardLogger
# from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor

# import dataset, network to train and metric to optimize
# from pytorch_forecasting import TemporalFusionTransformer
# from pytorch_forecasting.metrics import QuantileLoss, MultiLoss
from pytorch_forecasting import TemporalFusionTransformer, QuantileLoss, MultiLoss


class Model:
    def __init__(self, dataset, config):
        self.config = config
        self.dataset = dataset
        self.target_dates = self.gen_target_dates()
        self.model = self.gen_model()
        self.trainer = self.gen_trainer()

    def gen_target_dates(self):
        target_dates = pd.date_range(
            start=self.config.date,
            periods=self.config.dataset_config["max_prediction_length"],
        )
        target_dates = [date.strftime("%Y-%m-%d") for date in target_dates]
        return target_dates

    def gen_model(self):
        if isinstance(self.config.dataset_config['target'], list):
            if len(self.config.dataset_config['target']) >= 2:
                loss = MultiLoss([QuantileLoss(self.config.quantiles_ratio) for _ in self.config.dataset_config['target']])
                output_size = [len(self.config.quantiles_ratio) for _ in self.config.dataset_config['target']]
            else:
                loss = MultiLoss([QuantileLoss(self.config.quantiles_ratio)])
                output_size = len(self.config.quantiles_ratio)
        else:
            loss = QuantileLoss(self.config.quantiles_ratio)
            output_size = len(self.config.quantiles_ratio)

        tft = TemporalFusionTransformer.from_dataset(
            self.dataset.dataset,
            loss=loss,
            output_size=output_size,
            **self.config.model_config
        )
        return tft

    def gen_trainer(self):
        if self.config.use_gpu:
            trainer = pl.Trainer(
                **self.config.trainer_config,
                devices=1,
                accelerator="auto",
                strategy="auto",
                log_every_n_steps = 10,
            )
        else:
            trainer = pl.Trainer(
                **self.config.trainer_config,
                log_every_n_steps = 10,
            )
        return trainer

    def train(self):
        self.trainer.fit(
            self.model,
            train_dataloaders=self.dataset.train_dataloader,
            val_dataloaders=self.dataset.val_dataloader,
        )
        best_model_path = self.trainer.checkpoint_callback.best_model_path
        print(f"model save success, model path {best_model_path}")

    def valid(self):
        df = self.dataset.val_dataloader
        best_model_path = self.trainer.checkpoint_callback.best_model_path
        best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
        if self.config.valid:
            predictions, index = best_tft.predict(df, return_index=True)
            actuals = torch.cat([y[0] for x, y in iter(df)])
            print(f"predictions num: {predictions.sum(axis=0)}")
            print(f"actuals num: {actuals.sum(axis=0)}")
            wmape = (predictions - actuals).abs().sum(axis=0) / actuals.abs().sum(axis=0)
            print(f"prediction wmape: {wmape}")
        return True
