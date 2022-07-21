import logging

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pytorch_lightning as pl
import numpy as np
import pandas as pd
import os
from abc import ABC
#from experiments.configs.config import extern
from generate_simulated import load_mdp

pl.seed_everything(42)


class DatasetMDP(Dataset):
    """
    Create a simulated count_number data pipeline feeder.
    """

    @staticmethod
    def batch_to_data(batch):
        """
        :return:  x shape (seq_length, num_channels (chromosomes), num_sequences)
        """
        # From batch, get model input and output
        t_end = batch['t'+str(len(batch.keys())-2)]
        x = torch.reshape(t_end, (len(t_end), 1, 1))
        # Duplicate for major and minor sequences #TODO: could analyse further variation to see architecture effects
        x = torch.stack((x, x), dim=-1).squeeze(2)
        # add duplication of channels
        #x = torch.stack((x, x), dim=1).squeeze(2)

        y = batch["label"].squeeze(-1).long()
        # X shape: seq_len x chromosomes x channel
        return {'feature': x, 'label': y}

    def __init__(self, data: pd.DataFrame, label_encoder, weight_dict: dict = None):
        """
        """
        self.data_frame = data
        self.label_encoder = label_encoder
        self.weight_dict = weight_dict
        self.length = len(self.data_frame.index)                 # Get the number of samples from the dataframe

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get features
        filter_col = [col for col in self.data_frame if col.startswith('t')]         # Get the time point observations
        feature = self.data_frame.loc[self.data_frame.index[idx], filter_col]        # for next sample
        features_dict = {}
        for time_idx, f in enumerate(feature):                                       # Re-format each time point in loop
            features_dict["t"+str(time_idx)] = torch.Tensor([int(element) for element in f.split(',')])

        # Get label
        label = self.data_frame.loc[self.data_frame.index[idx], ['labels']][0]
        label_enc = list(self.label_encoder.classes_).index(label)
        label_dict = {"label": torch.Tensor([label_enc])}

        # Get all information in batch, so it can be wrapped
        batch = {**features_dict, **label_dict}

        return self.batch_to_data(batch)


class _DataModuleMDP(pl.LightningDataModule, ABC):
    """

    """
    @staticmethod
    def simulated():
        return True

    @property
    def n_classes(self):
        return len(self.label_encoder.classes_)

    @property
    def bases(self):
        if self.kernels is None:
            logging.warning("This is not a simulated example: True bases are not known")
        return self.kernels

    @property
    def n_bases(self):
        if self.kernels is None:
            logging.critical("This is not a simulated example: True bases are not known")
        return np.sum([len(k) for k in self.kernels])

    @property
    def seq_length(self):
        return self.length

    def __init__(self, batch_size=128, length=50, file_path=None):
        super(_DataModuleMDP, self).__init__()
        self.batch_size = batch_size
        self.length = length
        self.file_path = file_path

        self.train_df, self.val_df, self.test_df = None, None, None
        self.train_weights = None
        self.label_encoder = None
        self.training_set, self.validation_set, self.test_set = None, None, None
        self.training_sampler, self.train_shuffle = None, None
        self.kernels = None                        # If simulated model with known bases (TODO: or later learnt bases)

    def prepare_data(self):
        # Download, filter, format, split
        _, (self.train_df, self.val_df, self.test_df), self.weight_dict, self.label_encoder, self.kernels = \
            load_mdp(length=self.length, file_path=self.file_path)

    def setup(self, stage=None):
        #
        self.training_set = DatasetMDP(self.train_df, self.label_encoder, weight_dict=self.weight_dict)
        if self.weight_dict is not None:
            self.training_sampler = \
                WeightedRandomSampler(self.training_set.weights, len(self.training_set.weights), replacement=True)
            self.train_shuffle = False
        else:
            self.training_sampler = None
            self.train_shuffle = True

        self.validation_set = DatasetMDP(self.val_df, self.label_encoder)
        self.test_set = DatasetMDP(self.test_df, self.label_encoder)

    def train_dataloader(self):
        return DataLoader(
            sampler=self.training_sampler,
            dataset=self.training_set,
            batch_size=self.batch_size,
            num_workers=os.cpu_count(),
            shuffle=self.train_shuffle
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.validation_set,
            batch_size=self.batch_size,
            num_workers=os.cpu_count()
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_set,
            batch_size=self.batch_size,
            num_workers=os.cpu_count()
        )


#@extern
def DataModuleMDP(length=50, steps=5, n=10000, file_path=None, batch_size=256):
    # TODO: Fix kwargs to inherit from here
    # Wrapper around _DataModuleMDP, also with configuration decorator
    data_module = _DataModuleMDP(length=length,
                                 file_path=file_path,
                                 batch_size=batch_size)
    data_module.prepare_data()
    data_module.setup()

    return data_module


def main_test():
    # TODO: move to unit test
    data_module = DataModuleMDP()

    loader_list = {'train': data_module.train_dataloader(),
                   'test': data_module.test_dataloader(),
                   'validation': data_module.val_dataloader(),
                   }
    for key in loader_list:
        print(f'\n{key} set\n=============')
        for batch_idx, batch in enumerate(loader_list[key]):
            print(f'\nBatch {key} index {batch_idx}')
            print(f'Batch {batch.keys()}')
            print(f"label counts {torch.unique(batch['label'], return_counts=True)}")
            print(f"input shape {batch['feature'].shape}")   # N x length x regions x num_sequences
            print(f"output shape {batch['label'].shape}")


if __name__ == '__main__':

    main_test()
