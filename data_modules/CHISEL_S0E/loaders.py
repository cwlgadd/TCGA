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

pl.seed_everything(42)


class Load:
    """
    Class for loading data.
    """

    def __init__(self):
        self.load(os.path.dirname(os.path.abspath(__file__)) + r'/data/merged.csv')

    def __str__(self):
        s = f'CHISEL parser, subject 0, site E'
        return s

    def load(self, path):

        self.data_frame = pd.read_csv(path, index_col="CELL", usecols=[i for i in range(1,14)])
        # print(self.data_frame.head())
        # except:
        #     raise NotImplementedError
        #     self.data_frame = pd.read_csv(self.calls_path, usecols=[i for i in range(1,13)], index_col="CELL")
        #     self.data_frame_clones = pd.read_csv(self.clones_path, index_col="X.CELL", usecols=[i for i in range(1,4)])
        #
        #     self.data_frame["clone"] = np.nan
        #     for sample in list(self.data_frame.index.unique()):
        #         sample_clone = self.data_frame_clones[self.data_frame_clones.index == sample]["CLONE"][0]
        #         self.data_frame.loc[self.data_frame.index == sample, "clone"] = sample_clone
        #         # print(f"{sample} went to clone {sample_clone}")
        #
        #     # assert self.data_frame.isnull().values.any() is False
        #     self.data_frame.to_csv(path)
        #
        # #     clone_cols.append(self.data_frame_clones.iloc[self.data_frame_clones.index==row[0]]["CLONE"])

        return

    def train_test_split(self):
        # Split frame into training, validation, and test
        unique_samples = self.data_frame.index.unique()
        # print(f"{len(unique_samples)} unique samples")
        # print(unique_samples)

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
        return "\nCHISEL S0-E DataModule"

    def setup(self, stage=None):
        """

        @param stage:
        @return:
        """
        #
        self.label_encoder = preprocessing.LabelEncoder()
        self.label_encoder.fit_transform(self.data_frame.CLONE.unique())

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

            CNA_sequence = torch.ones((2, 22, chr_length))
            for row in subject_edge_info.iterrows():
                chrom = int(row[1]['X.CHR'][3:])     # 23 if row[1]['X.CHR'] in ['chrX', 'chrY'] else

                start_pos = int(np.floor(row[1]['START'] / true_chr_lengths[row[1]['X.CHR'][3:]] * chr_length))
                end_pos = int(np.floor(row[1]['END'] / true_chr_lengths[row[1]['X.CHR'][3:]] * chr_length))

                # Copy Number
                CN_STATE = row[1]['CN_STATE'].split("|")
                CNA_sequence[0, chrom-1, start_pos:end_pos] = int(CN_STATE[0])
                CNA_sequence[1, chrom-1, start_pos:end_pos] = int(CN_STATE[1])

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
        clone = subject_frame["CLONE"][0]
        label = list(self.label_encoder.classes_).index(clone)

        return {'feature': count_numbers ,
                'label': label,
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

        _tmp = self.data_frame.groupby('CELL').first()
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
