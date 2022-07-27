import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pytorch_lightning as pl
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import os
from abc import ABC
from .generate_simulated import SimulateMarkov

pl.seed_everything(42)


class DataModule(SimulateMarkov, pl.LightningDataModule, ABC):
    """

    """

    def __init__(self, steps, classes=2, channels=23, length=100, n=10000, n_kernels_per=30, n_kernels_shared=20,
                 path=None, batch_size=128):
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
        super(DataModule, self).__init__()
        SimulateMarkov.__init__(self,
                                classes=classes,
                                channels=channels,
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
        self.test_df, self.val_df = train_test_split(test_df, test_size=0.5)

        self.setup()

    def __str__(self):
        return str(self.training_set) + str(self.validation_set) + str(self.test_set)

    def setup(self, stage=None):
        self.training_set = MarkovDataset(self.train_df, self.label_encoder)
        self.test_set = MarkovDataset(self.test_df, self.label_encoder)
        self.validation_set = MarkovDataset(self.val_df, self.label_encoder)

    def train_dataloader(self):
        return DataLoader(
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

    def __init__(self, data: pd.DataFrame, label_encoder):
        """
        """
        self.data_frame = data
        self.label_encoder = label_encoder

        self.n_strands = len(self.data_frame.columns) - 1
        self.n_channels = len(self.data_frame.iloc[0]['strand 0'])
        self.seq_length = self.data_frame.iloc[0]['strand 0'][0].shape[0]

    def __len__(self):
        return len(self.data_frame.index)

    def __str__(self):
        s = f"\nsamples: {len(self.data_frame)}, " \
            f"strands: {self.n_strands}, " \
            f"channels {self.n_channels}, " \
            f"seq_length {self.seq_length}"
        return s

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get features
        feature = np.zeros((self.n_strands, self.n_channels, self.seq_length))
        for c in range(self.n_channels):
            for s in range(self.n_strands):
                feature[s, c, :] = self.data_frame.iloc[idx][f"strand {s}"][c]

        # Get label
        label = self.data_frame.loc[self.data_frame.index[idx], ['labels']][0]
        label_enc = list(self.label_encoder.classes_).index(label)

        return torch.Tensor(feature), torch.Tensor([label_enc])


def example_loader(steps=2, batch_size=256):

    data_module = DataModule(steps=steps, n=10000, classes=2, channels=10, n_kernels_per=3, n_kernels_shared=1,
                             batch_size=batch_size)
    print(data_module)
    loader_list = {'train': data_module.train_dataloader(),
                   'test': data_module.test_dataloader(),
                   'validation': data_module.val_dataloader(),
                   }
    for key in loader_list:
        print(f'\n{key} set\n=============')
        for batch_idx, batch in enumerate(loader_list[key]):
            print(f'\nBatch {key} index {batch_idx}')
            x, label = batch
            print(x.shape)
            print(label.shape)


if __name__ == '__main__':

    example_loader()
