import logging

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import os
from abc import ABC
from data.simulatedMarkov.generate_simulated import *

pl.seed_everything(42)


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


def debug_loader(plot=False, length=20):
    # TODO: turn into unit test

    data_module = MarkovDataModule(classes=2, steps=1, n=1000, length=length, n_class_bases=2, n_bases_shared=0)
    print(data_module)

    loader_list = {'train': data_module.train_dataloader(),
                   'test': data_module.test_dataloader(),
                   'validation': data_module.val_dataloader(),
                   }
    for key in loader_list:
        print(f'\n{key} set\n=============')
        for batch_idx, batch in enumerate(loader_list[key]):
            print(f'\nBatch {key} index {batch_idx}')
            print(f'Batch {batch.keys()}')
            print(f"Feature shape {batch['feature'].shape}, label shape {batch['label'].shape}")
            print(f"label counts {torch.unique(batch['label'], return_counts=True)}")
            print(np.unique(batch['feature'], axis=0).shape)

            if plot:
                import matplotlib.pyplot as plt
                strand, channel = 0, 0
                for l in [0, 1]:
                    samples = batch['feature'][batch['label'] == l, :][:, strand, channel, :]
                    for s in range(samples.shape[0]):
                        plt.scatter(np.linspace(1, length, length), samples[s, :] + 0.01*np.random.randn(samples[s,:].shape[0]))
                    plt.title(f'Label {l}')
                plt.show()


if __name__ == '__main__':

    debug_loader()
