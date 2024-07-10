from sklearn.model_selection import train_test_split as sk_split
from sklearn import preprocessing
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import column_or_1d

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


class LoadCHISEL_S0E:
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

        # Remove cells that werent assigned to a clonal structure (comment out to use all)
        self.data_frame = self.data_frame[self.data_frame.CLONE != "None"]
        # self.data_frame.drop(self.data_frame.loc[self.data_frame['CLONE'] == "None"].index, inplace=True)

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
        weight_dict = {}
        ntrain_unique_samples = len(train_df.index.unique())
        for cancer_id, group in train_df.groupby('CLONE'):
            unique_samples = len(group.index.unique()) / ntrain_unique_samples
            if unique_samples > 0:
                weight_dict[cancer_id] = 1 / unique_samples

        return (train_df, test_df, val_df), weight_dict


class DataModule(pl.LightningDataModule, ABC):
    """

    """

    def __init__(self, batch_size=128, chr_length=256, stack=False, custom_edges=None,
                 sampler=False):
        """

        @param sampler:
            Boolean flag whether we use a weighted random sampler
        """
        super().__init__()
        self.prepare_data_per_node = False
        self.train_set, self.test_set, self.validation_set = None, None, None

        self.batch_size = batch_size
        self.chr_length = chr_length
        self.stack = stack
        self.edges2 = custom_edges
        self.train_sampler, self.train_shuffle = None, True
        # self.label_encoder = None 
        self.sampler = sampler
        self.W = self.chr_length * 22 if self.stack else self.chr_length
        self.C = 2 if self.stack else 2 * 22

        self.prepare_data()
        self.setup()
        
    def __str__(self):
        return "\nCHISEL S0-E DataModule"

    def prepare_data(self, stage=None):
        self.dataset = LoadCHISEL_S0E()


    def setup(self, stage=None):
        """

        @param stage:
        @return:
        """
        #
        self.label_encoder = OrderedLabelEncoder()     #preprocessing.LabelEncoder()
        label_order = ["Clone5", "Clone63", "Clone156", "Clone172", "Clone199", "Clone241", "None"]
        # label_order = self.data_frame.CLONE.unique()
        self.label_encoder.fit_transform(label_order)

        (self.train_df, self.val_df, self.test_df), weight_dict = self.dataset.train_test_split()
        self.weight_dict = weight_dict if self.sampler else None

        self.train_set = Dataset(self.train_df, self.label_encoder, self.chr_length,
                                 stack=self.stack,
                                 weight_dict=self.weight_dict,
                                 custom_edges2seq=self.edges2)

        if self.weight_dict is not None:
            print(f"Using weight dictionary")
            self.train_sampler = WeightedRandomSampler(self.train_set.weights,
                                                       len(self.train_set.weights),
                                                       replacement=True)
            self.train_shuffle = False

        self.test_set = Dataset(self.test_df, self.label_encoder, self.chr_length,
                                stack=self.stack, custom_edges2seq=self.edges2)
        self.validation_set = Dataset(self.val_df, self.label_encoder, self.chr_length,
                                      stack=self.stack, custom_edges2seq=self.edges2)

    def train_dataloader(self):
        return DataLoader(
            sampler=self.train_sampler,
            dataset=self.train_set,
            batch_size=self.batch_size,
            num_workers=np.min((8,os.cpu_count())),
            shuffle=self.train_shuffle
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.validation_set,
            batch_size=self.batch_size,
            num_workers=np.min((8,os.cpu_count())),
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_set,
            batch_size=self.batch_size,
            num_workers=np.min((8,os.cpu_count())),
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
                CNA_sequence[0, chrom-1, start_pos:end_pos] = int(CN_STATE[0])      # np.min((, 5))
                CNA_sequence[1, chrom-1, start_pos:end_pos] = int(CN_STATE[1])      # np.min((, 5))

        else:
            # TODO: Above assumes each chromosome has equal length - implement alternative with zero-padding
            raise NotImplementedError

        return CNA_sequence

    def default_df2data(self, subject_frame, stack):
        """
        :return:  x shape (seq_length, num_channels, num_sequences)
        """
        subject_frame.reset_index(drop=True, inplace=True)
        # print(subject_frame.columns)

        # Get the count numbers
        count_numbers = self.edges2seq(subject_frame)

        # Get clonal labels
        clone = subject_frame["CLONE"][0]
        label = list(self.label_encoder.classes_).index(clone)

        if stack:
            # Stack chromosomes
            count_numbers = count_numbers.view(count_numbers.size(0), -1)
        else:
            # channelise chromosomes
            count_numbers = count_numbers.view(count_numbers.size(0) * count_numbers.size(1), -1)

        return {'CNA': count_numbers,
                'label': torch.tensor(label),
                }

    def __init__(self, data: pd.DataFrame, label_encoder, chr_length=256, stack=False, weight_dict: dict = None,
                 custom_df2data=None, custom_edges2seq=None):
        """

        @param data:
        @param label_encoder:
        @param weight_dict:
        @param custom_df2data:           Custom method to wrap the DataLoader output
        @param custom_edges2seq:         Custom method to wrap for feature output from condensed edge representation
        """
        self.chr_length = chr_length
        self.stack = stack
        self.data_frame = data
        self.label_encoder = label_encoder
        
        # custom wrappers
        self.df2data = custom_df2data if custom_df2data is not None else self.default_df2data
        self.edges2seq = custom_edges2seq if custom_edges2seq is not None else self.default_edges2seq

        _tmp = self.data_frame.groupby('CELL').first()
        self.IDs = [row[0] for row in _tmp.iterrows()]

        if weight_dict is not None:
            # for row in _tmp.iterrows():
                # print(row)
            self.weights = [weight_dict[row[1].CLONE] for row in _tmp.iterrows()]

    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        subject_frame = self.data_frame.loc[[self.IDs[idx]]]
        return self.df2data(subject_frame, self.stack)


# Additional functionality - wrapper for sklearn's label encoder so we can define the class order
def ordered_encode_python(values, uniques=None, encode=False):
    # only used in _encode below, see docstring there for details
    if uniques is None:
        uniques = list(dict.fromkeys(values))
        uniques = np.array(uniques, dtype=values.dtype)
    if encode:
        table = {val: i for i, val in enumerate(uniques)}
        try:
            encoded = np.array([table[v] for v in values])
        except KeyError as e:
            raise ValueError("y contains previously unseen labels: %s"
                             % str(e))
        return uniques, encoded
    else:
        return uniques

class OrderedLabelEncoder(LabelEncoder):
    def fit(self, y):
        y = column_or_1d(y, warn=True)
        self.classes_ = ordered_encode_python(y)

    def fit_transform(self, y):
        y = column_or_1d(y, warn=True)
        self.classes_, y = ordered_encode_python(y, encode=True)
        return y
