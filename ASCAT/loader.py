import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pytorch_lightning as pl
import numpy as np
import pandas as pd
import os
from abc import ABC
#from experiments.configs.config import extern
from ascat import load_ascat

pl.seed_everything(42)


class DatasetASCAT(Dataset):
    """
    From dataframe create count_number data pipeline feeder, formatted for BDLSTM model.
    We keep data in the condensed (startpos, endpos) format for memory efficiency at the cost of some minor overhead.
    """

    def get_series(self, sequence):
        """
        Helper function to convert collections of (startpos, endpos) into full (likely coarse grained) sequences. This
        is called during __getitem__ to avoid storing many large vectors.
        """
        nmajor, nminor = np.ones((self.length,)), np.ones((self.length,))
        start_pos = sequence['startpos'].to_numpy()
        end_pos = sequence['endpos'].to_numpy()

        for idx, (start, end) in enumerate(zip(start_pos, end_pos)):
            # Get start and end points for indexing. If last segment, always round up 'endpos' due to added padding zone
            start = int(round(start))
            if idx == start_pos.shape[0] - 1:
                end = int(np.ceil(end))
            else:
                end = int(round(end))

            nmajor[start:end] = sequence.nmajor.to_numpy()[idx]
            nminor[start:end] = sequence.nminor.to_numpy()[idx]

        return nmajor, nminor

    @staticmethod
    def batch_to_data(batch):
        """
        :return:  x shape (seq_length, num_channels, num_sequences)
        """
        # From batch, get model input and output
        x = torch.stack((batch["nmajor"], batch["nminor"]), dim=-1).squeeze(2)
        y = batch["cancer_type"].squeeze(-1).long()
        return {'feature': x, 'label': y}

    def __init__(self, data: pd.DataFrame, label_encoder, weight_dict: dict = None):
        """
        """
        # TODO: transform on samples needed?

        self.data_frame = data
        self.label_encoder = label_encoder

        # sequence length
        # TODO: move this so its shared for train/val/test frames, or some other way to make it more robust
        self.length = int(np.ceil(self.data_frame['endpos'].max()))

        # Extract and format count number data from data frame. Separate by chromosome for potential model channels
        # TODO: This can't be vectorised as we need each sample's chromosome sequence to have its own structure,
        #  ... but can make indexing/code structure here much more logical and clean!
        chrom_sequences = []
        for chromosome_id, group_c in self.data_frame.groupby('chr'):

            sequences = []                      # List of feature series and labels for one chromosome
            weights = []
            for patient_id, group_cp in group_c.groupby('sample'):

                # The label information
                cancer_type = group_cp['cancer_type'].to_numpy()[0]
                cancer_type_enc = list(self.label_encoder.classes_).index(cancer_type)
                label = [cancer_type_enc,
                         group_cp['wgd'].to_numpy()[0],
                         group_cp['gi'].to_numpy()[0],
                         ]
                if weight_dict is not None:
                    weights.append(weight_dict[cancer_type])

                # The information required to reconstruct the sequence of count numbers
                sequence_features = group_cp[['startpos', 'endpos', 'nmajor', 'nminor']]

                # Store for __getitem__
                sequences.append((sequence_features, label))

            chrom_sequences.append(sequences)

        # The sequences along each chromosome
        self.sequences = chrom_sequences         # List of chromosomes[list of samples[tuple of features and label]]
        self.weights = weights

    def __len__(self):
        return len(self.sequences[0])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Collect sample
        chr_seq_maj, chr_seq_minor = [], []
        for chrom in range(len(self.sequences)):
            sequence, label = self.sequences[chrom][idx]
            # Convert (startpos, endpos) format into full sequence
            sequence_major, sequence_minor = self.get_series(sequence)
            chr_seq_maj.append(sequence_major)
            chr_seq_minor.append(sequence_minor)

        seq_maj = np.stack(chr_seq_maj, axis=1)
        seq_min = np.stack(chr_seq_minor, axis=1)

        # Get all information in batch, so it can be wrapped
        batch = dict(nmajor=torch.Tensor(seq_maj),             # Sequence 'length' x Number of chromosomes
                     nminor=torch.Tensor(seq_min),             # Sequence 'length' x Number of chromosomes
                     cancer_type=torch.Tensor([label[0]]),     # Scalar
                     wgd=torch.Tensor([label[1]]),
                     gi=torch.Tensor([label[2]]),
                     )

        return self.batch_to_data(batch)


class _DataModuleASCAT(pl.LightningDataModule, ABC):
    """

    """
    @staticmethod
    def simulated():
        return False

    @property
    def n_classes(self):
        return len(self.label_encoder.classes_)

    @property
    def seq_length(self):
        return self.length

    def __init__(self, batch_size=128, length=500, file_path=None, cancer_types=None, wgd=None, chromosomes=None):
        super(_DataModuleASCAT, self).__init__()
        self.batch_size = batch_size
        self.length = length
        self.file_path = file_path
        self.cancer_types = cancer_types
        self.wgd = wgd
        self.chromosomes = chromosomes

        self.train_df, self.val_df, self.test_df = None, None, None
        self.train_weights = None
        self.label_encoder = None
        self.training_set, self.validation_set, self.test_set = None, None, None
        self.training_sampler, self.train_shuffle = None, None

    def prepare_data(self):
        # Download, filter, format, split
        _, (self.train_df, self.val_df, self.test_df), self.weight_dict, self.label_encoder = \
            load_ascat(length=self.length, file_path=self.file_path,
                       cancer_types=self.cancer_types, wgd=self.wgd, chromosomes=self.chromosomes)

    def setup(self, stage=None):
        #
        self.training_set = DatasetASCAT(self.train_df, self.label_encoder, weight_dict=self.weight_dict)
        if self.weight_dict is not None:
            self.training_sampler = \
                WeightedRandomSampler(self.training_set.weights, len(self.training_set.weights), replacement=True)
            self.train_shuffle = False
        else:
            self.training_sampler = None
            self.train_shuffle = True

        self.validation_set = DatasetASCAT(self.val_df, self.label_encoder)
        self.test_set = DatasetASCAT(self.test_df, self.label_encoder)

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


# @extern
def DataModuleASCAT(length=250, file_path=None, cancer_types=None, wgd=None, chromosomes=None, batch_size=256):
    # Wrapper around _DataModuleASCAT, also with configuration decorator
    data_module = _DataModuleASCAT(length=length,
                                   file_path=file_path,
                                   cancer_types=cancer_types,
                                   wgd=wgd,
                                   chromosomes=chromosomes,
                                   batch_size=batch_size)
    data_module.prepare_data()
    data_module.setup()

    return data_module


def main_test():
    # TODO: move to unit test
    data_module = DataModuleASCAT()

    loader_list = {'train': data_module.train_dataloader(),
                   'test': data_module.test_dataloader(),
                   'validation': data_module.val_dataloader(),
                   }
    for key in loader_list:
        print(f'{key} set\n=============')
        for batch_idx, batch in enumerate(loader_list[key]):
            print(f'\nBatch {key} index {batch_idx}')
            print(f"cancer type counts {torch.unique(batch['label'], return_counts=True)}")
            print(f"input shape {batch['feature'].shape}")
            print(f"output shape {batch['label'].shape}")


if __name__ == '__main__':

    main_test()
