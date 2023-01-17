# Methods for loading and parsing the simulated version of the ascat dataset into a dataframe.
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn import preprocessing
from sklearn.model_selection import train_test_split as sk_split
import pytorch_lightning as pl
import os
from abc import ABC
import pandas as pd
import numpy as np
import logging


class SimulateMarkov:
    """
    Create or load simulated version of ASCAT count number data.
    """

    def make_basis(self):
        """ Sample beginning pos and length of a transition, of all +1
        """
        # Maximum width should not exceed 2/3 of length, and should not be lower than
        _max_width = np.floor(2 * self.length / 3)

        delta_state = np.zeros(self.length)
        start = np.random.randint(0, self.length - (_max_width - 1))  # Sample (from 0 to length-5)
        width = np.random.randint(1, (_max_width + 1))  # Number of elements that change (up to _max_width)
        delta_state[start:start + width] += 1
        return delta_state

    @property
    def last_state(self):
        return self.trajectories[:, -1, :]

    @property
    def frame(self):
        # Creates a new frame every call, #TODO
        # This is what is read into the loader
        d = {'features': list(self.last_state), 'labels': self.labels}
        return pd.DataFrame(data=d)

    def __init__(self, classes=2, length=100, n=50000, n_class_bases=30, n_bases_shared=20, path=None, init_steps=1):
        assert n_bases_shared < n_class_bases
        self.classes = classes
        self.length = length
        self.n = n
        self.n_kernels = n_class_bases
        self.n_kernels = n_bases_shared
        self.path = path

        # Create the shared kernels/bases
        _shared_bases = [self.make_basis() for _ in range(n_bases_shared)]
        # Create the remaining, independent, kernels/bases of each class
        self.bases = [
            [self.make_basis() for _ in range(n_class_bases - n_bases_shared)] + _shared_bases
            for _ in range(classes)
        ]

        # Begin all samples at the 1 count at every locus
        self.trajectories = np.ones((n, 1, length))     # N x T x D
        # And randomly assign each sample to a different class with equal probability
        self.labels = np.random.choice([i for i in range(classes)], size=n, p=[1./classes for _ in range(classes)])

        if init_steps > 0:
            self(init_steps)

    def __str__(self):
        s = "SimulateMarkov class summary\n==========================="
        for idx_c, c in enumerate(self.bases):
            s += f"\nClass {idx_c} has bases:"
            s += f"\n{np.vstack(c)}"

        combinations = np.unique(self.last_state, axis=0)
        s += f"\n ... giving (end of trajectory, steps={self.trajectories.shape[1]-1}) " \
             f"{combinations.shape[0]} combinations:\n{combinations}"
        return s

    def __call__(self, steps=1):
        """ Sample through the Markov Process
        """
        s_t = np.zeros((self.n, steps + 1, self.length))
        s_t[:, 0, :] = self.last_state

        for step in range(steps):
            delta = np.zeros_like(self.last_state)
            for label in range(self.classes):
                n_basis_in_class = len(self.bases[label])
                idx_next_basis = np.random.choice([i for i in range(n_basis_in_class)], size=sum(self.labels == label),
                                                  p=[1. / n_basis_in_class for _ in range(n_basis_in_class)]
                                                  )
                delta_class = np.vstack([self.bases[label][idx] for idx in idx_next_basis])
                delta[self.labels == label] = delta_class
            s_t[:, step + 1, :] = s_t[:, step, :] + delta

        self.trajectories = np.concatenate((self.trajectories, s_t[:, 1:, :]), axis=1)

        return self.trajectories


class MarkovDataModule(SimulateMarkov, pl.LightningDataModule, ABC):
    """

    """
    @property
    def num_cancer_types(self):
        return len(self.label_encoder.classes_)

    def __init__(self, steps, classes=2, length=100, n=1000, n_class_bases=2, n_bases_shared=0, path=None,
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

        self.batch_size = batch_size
        self.training_set, self.test_set, self.validation_set = None, None, None

        # Define simulated set, and run process forward
        SimulateMarkov.__init__(self,
                                classes=classes,
                                length=length,
                                n=n,
                                n_class_bases=n_class_bases,
                                n_bases_shared=n_bases_shared,
                                path=path,
                                init_steps=steps)

        _df = self.frame

        # Encode remaining type labels, so they can be used by the model later
        self.label_encoder = preprocessing.LabelEncoder()
        self.label_encoder.fit_transform(_df.labels.unique())

        # Split frame into training, validation, and test
        self.train_df, test_df = sk_split(_df, test_size=0.2)
        self.test_df, self.val_df = sk_split(test_df, test_size=0.2)

        self.setup()

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
        sample = self.data_frame.loc[self.data_frame.index[idx]]
        feature = np.tile(sample['features'], (2, 1, 1))          # Just duplicate 2nd strand,  for now

        # Get label
        label = self.data_frame.loc[self.data_frame.index[idx], ['labels']][0]
        label_enc = list(self.label_encoder.classes_).index(label)

        batch = {"feature": torch.tensor(feature, dtype=torch.float),
                 "label": torch.tensor(label_enc)}
        return batch


