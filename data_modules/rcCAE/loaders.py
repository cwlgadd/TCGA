from sklearn.model_selection import train_test_split as sk_split
from sklearn import preprocessing
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
import pytorch_lightning as pl
import os
import re
from abc import ABC
from tqdm import tqdm
import numpy as np
import pandas as pd
import pyarrow.feather as feather

from TCGA.data_modules.utils.helpers import get_chr_base_pair_lengths as chr_lengths

# pl.seed_everything(42)


class Load:
    """
    Class for loading data.
    """

    def __init__(self):
        self.path = os.path.dirname(os.path.abspath(__file__)) + r'/data/breast_tissue_E_2k_node_cnv_calls.csv'
        self.data_frame = self.load(self.path)

    def __str__(self):
        s = f'rcCAE parser (from rc-convolutional auto encoder paper)'
        return s

    def load(self, path):
        self.data_frame = pd.read_csv(path, usecols=[i for i in range(1,7)], index_col="sample")
        return self.data_frame

    def train_test_split(self):
        # Split frame into training, validation, and test
        unique_samples = self.data_frame.index.unique()
        print(f"{len(unique_samples)} unique samples")

        train_labels, test_labels = sk_split(unique_samples, test_size=0.2)
        test_labels, val_labels = sk_split(test_labels, test_size=0.5)
        train_df = self.data_frame[self.data_frame.index.isin(train_labels)]
        test_df = self.data_frame[self.data_frame.index.isin(test_labels)]
        val_df = self.data_frame[self.data_frame.index.isin(val_labels)]
        # assert len(train_df.cancer_type.unique()) == len(self.data_frame.cancer_type.unique()),\
        #     'Check all labels are represented in training set'

        # Random sampler weights
        weight_dict = None

        return (train_df, test_df, val_df), weight_dict


class DataModule(Load, pl.LightningDataModule, ABC):
    """

    """

    def __init__(self, batch_size=128, custom_edges=None):
        """

        @param batch_size:
        """
        super(DataModule, self).__init__()

        self.batch_size = batch_size
        self.edges2 = custom_edges
        self.train_set, self.test_set, self.validation_set = None, None, None
        self.train_sampler, self.train_shuffle = None, None
        # self.label_encoder = None

        self.setup()
        self.W = self.train_set.chr_length

    def __str__(self):
        s = "\nrc-CAE DataModule"
        # for df, data_set in zip([self.train_df, self.val_df, self.test_df], ["Train", "Validation", "Test"]):
        #     df = df.groupby('sample').first()
        #     s += f"\n\t {data_set}"
        #     for i, j in zip(df['cancer_type'].value_counts().keys(), df['cancer_type'].value_counts().values):
        #         if j > 0:
        #             s += f"\n\t\t {i.ljust(8)}: {j}"
        return s

    def setup(self, stage=None):
        """

        @param stage:
        @return:
        """
        #
        self.label_encoder = None  # preprocessing.LabelEncoder()
        # self.label_encoder.fit_transform(self.data_frame.cancer_type.unique())

        (self.train_df, self.val_df, self.test_df), self.weight_dict = self.train_test_split()

        self.train_set = Dataset(self.train_df, self.label_encoder,
                                 weight_dict=self.weight_dict,
                                 custom_edges2seq=self.edges2)

        # if self.weight_dict is not None:
        #     self.train_sampler = WeightedRandomSampler(self.train_set.weights,
        #                                                len(self.train_set.weights),
        #                                                replacement=True)
        #     self.train_shuffle = False

        self.test_set = Dataset(self.test_df, self.label_encoder, custom_edges2seq=self.edges2)
        self.validation_set = Dataset(self.val_df, self.label_encoder, custom_edges2seq=self.edges2)

    def train_dataloader(self):
        return DataLoader(
            sampler=self.train_sampler,
            dataset=self.train_set,
            batch_size=self.batch_size,
            num_workers=os.cpu_count(),
            shuffle=self.train_shuffle
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.validation_set,
            batch_size=self.batch_size,
            num_workers=os.cpu_count(),
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_set,
            batch_size=self.batch_size,
            num_workers=os.cpu_count()
        )


class Dataset(Dataset):
    """
    Create ASCAT pipeline data feeder.

    We keep data in the condensed (startpos, endpos) format for memory efficiency at the cost of some minor overhead.
    """

    def default_edges2seq(self, subject_edge_info, equal_chr_length=True):
        """
        Helper function to convert collections of (startpos, endpos) into down-sampled sequences. This is called during
        __getitem__ to avoid storing many large vectors.
        """
        true_chr_lengths = chr_lengths()

        if equal_chr_length is True:
            chr_length = self.chr_length

            CNA_sequence = torch.ones((1, 23, chr_length))
            for row in subject_edge_info.iterrows():
                chrom = 23 if row[1]['chr'] in ['chrX', 'chrY'] else int(row[1]['chr'][3:])

                start_pos = int(np.floor(row[1]['startpos'] / true_chr_lengths[row[1]['chr'][3:]] * chr_length))
                end_pos = int(np.floor(row[1]['endpos'] / true_chr_lengths[row[1]['chr'][3:]] * chr_length))

                # Copy Number
                CNA_sequence[0, chrom-1, start_pos:end_pos] = row[1]['copy_number']

        else:
            # TODO: Above assumes each chromosome has equal length - implement alternative with zero-padding
            raise NotImplementedError

        return CNA_sequence

    def default_df2data(self, subject_frame):
        """
        :return:  x shape (seq_length, num_channels, num_sequences)
        """
        subject_frame.reset_index(drop=True, inplace=True)
        # print(subject_frame.columns)

        # Get the count numbers
        count_numbers = self.edges2seq(subject_frame)

        # Get labels
        # cancer_name = subject_frame["cancer_type"][0]
        # label = list(self.label_encoder.classes_).index(cancer_name)
        # label = self.label_encoder.transform([cancer_name])
        
        return {'feature': count_numbers,
                'label': torch.tensor(-1),
                }

    def __init__(self, data: pd.DataFrame, label_encoder, weight_dict: dict = None,
                 custom_df2data=None, custom_edges2seq=None):
        """

        @param data:
        @param label_encoder:
        @param weight_dict:
        @param custom_df2data:           Custom method to wrap the DataLoader output
        @param custom_edges2seq:         Custom method to wrap for feature output from condensed edge representation
        """
        self.chr_length = 256
        self.data_frame = data
        self.label_encoder = label_encoder
        
        # custom wrappers
        self.df2data = custom_df2data if custom_df2data is not None else self.default_df2data
        self.edges2seq = custom_edges2seq if custom_edges2seq is not None else self.default_edges2seq

        _tmp = self.data_frame.groupby('sample').first()
        self.IDs = [row[0] for row in _tmp.iterrows()]
        # if weight_dict is not None:
        #     self.weights = [weight_dict[row[1].cancer_type] for row in _tmp.iterrows()]

    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        subject_frame = self.data_frame.loc[[self.IDs[idx]]]
        return self.df2data(subject_frame)
