import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pytorch_lightning as pl
import numpy as np
import pandas as pd
import os
from abc import ABC
from data.ASCAT.ascat import *
from data.helpers import get_chr_base_pair_lengths as chr_lengths

pl.seed_everything(42)


class ASCATDataModule(ASCAT, pl.LightningDataModule, ABC):
    """

    """

    def __init__(self, batch_size=128, file_path=None, cancer_types=None, wgd=None):
        """

        @param batch_size:
        @param file_path:
        @param cancer_types:
        @param wgd:
        """
        if file_path is None:
            file_path = os.path.dirname(os.path.abspath(__file__)) + '/ascat.pkl'

        super(ASCATDataModule, self).__init__(path=file_path, cancer_types=cancer_types, wgd=wgd)

        self.batch_size = batch_size
        self.train_set, self.test_set, self.validation_set = None, None, None
        self.train_sampler, self.train_shuffle = None, False
        self.label_encoder = None

        self.setup()

    def setup(self, stage=None):
        """

        @param stage:
        @return:
        """
        #
        self.label_encoder = preprocessing.LabelEncoder()
        self.label_encoder.fit_transform(self.data_frame.cancer_type.unique())

        (self.train_df, self.val_df, self.test_df), self.weight_dict = self.train_test_split()

        self.train_set = ASCATDataset(self.train_df, self.label_encoder, weight_dict=self.weight_dict)
        if self.weight_dict is not None:
            self.train_sampler = WeightedRandomSampler(self.train_set.weights,
                                                       len(self.train_set.weights),
                                                       replacement=True)
            self.train_shuffle = False

        self.test_set = ASCATDataset(self.test_df, self.label_encoder)
        self.validation_set = ASCATDataset(self.val_df, self.label_encoder)

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
            num_workers=os.cpu_count()
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_set,
            batch_size=self.batch_size,
            num_workers=os.cpu_count()
        )


class ASCATDataset(Dataset):
    """
    Create ASCAT pipeline data feeder.

    We keep data in the condensed (startpos, endpos) format for memory efficiency at the cost of some minor overhead.
    """

    def edges2seq(self, subject_edge_info, equal_chr_length=True):
        """
        Helper function to convert collections of (startpos, endpos) into down-sampled sequences. This is called during
        __getitem__ to avoid storing many large vectors.
        """
        true_chr_lengths = chr_lengths()

        if equal_chr_length is True:
            chr_length = 1000

            CNA_sequence = torch.ones((2, 23, chr_length))
            for row in subject_edge_info.iterrows():
                chrom = 23 if row[1]['chr'] in ['X', 'Y'] else int(row[1]['chr'])

                start_pos = int(np.floor(row[1]['startpos'] / true_chr_lengths[row[1]['chr']] * chr_length))
                end_pos = int(np.floor(row[1]['endpos'] / true_chr_lengths[row[1]['chr']] * chr_length))

                # Major strand
                CNA_sequence[0, chrom-1, start_pos:end_pos] = row[1]['nMajor']
                # Minor strand
                CNA_sequence[1, chrom-1, start_pos:end_pos] = row[1]['nMinor']

        else:
            # TODO: Shouldn't assume each chromosome has equal length in models - implement this later
            raise NotImplementedError

        return CNA_sequence

    def df2data(self, subject_frame):
        """
        :return:  x shape (seq_length, num_channels, num_sequences)
        """
        subject_frame.reset_index(drop=True, inplace=True)

        # Get the columns relevant for the count number sequences
        subject_edge_info = subject_frame[['startpos', 'endpos', 'nMajor', 'nMinor', 'chr']]
        count_numbers = self.edges2seq(subject_edge_info)

        # Get label
        cancer_name = subject_frame["cancer_type"][0]
        label = list(self.label_encoder.classes_).index(cancer_name)
        # label = self.label_encoder.transform([cancer_name])

        return {'feature': count_numbers,
                'label': torch.tensor(label),
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
        self.data_frame = data
        self.label_encoder = label_encoder
        if custom_df2data is not None:
            self.df2data = custom_df2data
        if custom_edges2seq is not None:
            self.edges2seq = custom_edges2seq

        _tmp = self.data_frame.groupby('ID').first()
        self.IDs = [row[0] for row in _tmp.iterrows()]
        if weight_dict is not None:
            self.weights = [weight_dict[row[1].cancer_type] for row in _tmp.iterrows()]

    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        subject_frame = self.data_frame.loc[[self.IDs[idx]]]
        return self.df2data(subject_frame)


def debug_loader():
    # TODO: turn into unit test

    data_module = ASCATDataModule(wgd=['0'], cancer_types=['ACC', 'BRCA'])
    print(data_module)
    print(data_module.weight_dict)
    print(data_module.label_encoder.classes_)

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
            # for s in range(batch['feature'].shape[0]):
            #     print(f"input  {batch['feature'][s, 0, 0, :]}")   # N x length x regions x num_sequences
            #     print(f"output  {batch['label'][s]}")


if __name__ == '__main__':

    debug_loader()
