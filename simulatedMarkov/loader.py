import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pytorch_lightning as pl
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import os
from abc import ABC
from generate_simulated import SimulateMarkov

pl.seed_everything(42)


class MarkovDataModule(SimulateMarkov, pl.LightningDataModule, ABC):
    """

    """

    def __init__(self, steps, classes=2, length=100, n=50000, n_kernels_per=30, n_kernels_shared=20, path=None,
                 batch_size=128):
        """

        @param steps:               Number of steps to Markov Chain
        @param classes:             Number of different Markov Chains
        @param length:              Dimension of each state in Markov Chain
        @param n:                   Number of samples
        @param n_kernels_per:       Number of kernel transitions per Markov Chain
        @param n_kernels_shared:    Number of those kernel transitions that are shared between chains
        @param path:                Path for saving
        @param batch_size:          Batch size to load data into model
        """
        super(MarkovDataModule, self).__init__()
        SimulateMarkov.__init__(self,
                                classes=classes,
                                length=length,
                                n=n,
                                n_kernels_per=n_kernels_per,
                                n_kernels_shared=n_kernels_shared,
                                path=path
                                )

        self.batch_size = batch_size
        self.training_set, self.test_set, self.validation_set = None, None, None

        # Run Markov Process forward
        self(steps=steps)
        data_frame = self.make_data_frame()

        # Encode remaining type labels, so they can be used by the model later
        self.label_encoder = preprocessing.LabelEncoder()
        self.label_encoder.fit_transform(data_frame.labels.unique())

        # Split frame into training, validation, and test
        self.train_df, test_df = train_test_split(data_frame, test_size=0.2)
        self.test_df, self.val_df = train_test_split(data_frame, test_size=0.2)

    def setup(self, stage=None):
        self.training_set = MarkovDataset(self.train_df, self.label_encoder)
        self.test_set = MarkovDataset(self.test_df, self.label_encoder)
        self.validation_set = MarkovDataset(self.val_df, self.label_encoder)

    def train_dataloader(self):
        return DataLoader(
            sampler=None,
            dataset=self.training_set,
            batch_size=self.batch_size,
            num_workers=os.cpu_count(),
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.validation_set,
            batch_size=self.batch_size,
            num_workers=os.cpu_count(),
            shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_set,
            batch_size=self.batch_size,
            num_workers=os.cpu_count(),
            shuffle=False
        )


class MarkovDataset(Dataset):

    @staticmethod
    def make_channels(sequence):
        #TODO: unused

        # Just duplicate for now
        return np.tile(sequence, (1, 2))

    def __init__(self, data: pd.DataFrame, label_encoder):
        """
        """
        self.data_frame = data
        self.label_encoder = label_encoder
        self.n = len(self.data_frame.index)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get features
        filter_col = [col for col in self.data_frame if col.startswith('gene')]      # Get the time point observations
        feature = self.data_frame.loc[self.data_frame.index[idx], filter_col]        # for next sample

        # Get label
        label = self.data_frame.loc[self.data_frame.index[idx], ['labels']][0]
        label_enc = list(self.label_encoder.classes_).index(label)

        batch = {"feature": torch.Tensor(feature),
                 "label": torch.Tensor([label_enc])}
        return batch


def example_loader(steps=5, batch_size=256):

    data_module = MarkovDataModule(steps=steps, batch_size=batch_size)
    data_module.setup()

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

    example_loader()
