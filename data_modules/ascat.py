# Methods for loading and parsing the ascat data set into a dataframe
#
from sklearn.model_selection import train_test_split as sk_split
from sklearn import preprocessing
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pytorch_lightning as pl
import os
import re
from abc import ABC
from tqdm import tqdm
import numpy as np
import pandas as pd

from .utils.helpers import get_chr_base_pair_lengths as chr_lengths

pl.seed_everything(42)


class LoadASCAT:
    """
    Class for loading the ascat data
    """

    @property
    def num_cancer_types(self):
        return len(self.data_frame['cancer_type'].unique())

    def __init__(self, path=None, cancer_types=None, wgd=None):
        self.path = path
        self.filters = {'cancer': cancer_types, 'WGD': wgd}
        self._data_path = os.path.dirname(os.path.abspath(__file__)) + r'/data/ascat/ReleasedData/TCGA_SNP6_hg19'

        try:
            self.data_frame = self.load(self.path)
        except:
            print(f"Loading from submodule into pandas data frame.")
            self.parse_files()
            if path is not None:
                self.save(self.path)

        self.apply_filter(cancer_types=cancer_types, wgd=wgd)

    def __str__(self):
        s = f'Allele-Specific Copy Number Analysis of Tumors (ASCAT) parser, with filters {self.filters}'
        return s

    def load(self, path):
        self.data_frame = pd.read_pickle(path)
        return self.data_frame

    def save(self, path):
        pd.to_pickle(self.data_frame, path)

    def parse_files(self):
        """ Convert the numerous released data files into a single, condensed, dataframe so we don't have to load files
            whilst training our PyTorch model.

        @return: pandas data frame containing the condensed representation of the ASCAT data (i.e. with start/end pos)
        """

        # Load in the label information -> patient ID, cancer type, WGD, gi
        label_frame = pd.read_csv(self._data_path + r'/summary.ascatv3TCGA.penalty70.hg19.tsv',
                                  delimiter='\t',
                                  index_col='name'
                                  )

        # Load in the count number data
        all_subjects = []
        # Loop over segments files (each file belongs to one patient)
        for idx_seg, entry in tqdm(enumerate(os.scandir(self._data_path + '/segments/')),
                                   desc='Parsing count number data from files',
                                   total=len(label_frame.index)):

            # Extract patient identifier from segments file, cross reference against label file, and store labels
            if (entry.path.endswith(r".segments.txt")) and entry.is_file():
                try:
                    # Get patient identifier
                    subject_id = re.search(r'segments/(.+?).segments.txt', entry.path).group(1)
                    # print(subject_id)

                    # Load segment file
                    sample_frame = pd.read_csv(entry.path, sep='\t', index_col='sample')
                    assert len(sample_frame.index) > 0, f'Skipping empty segment file {subject_id}'

                    # Find label corresponding to patient identifier
                    subject_labels = label_frame.loc[[subject_id]]
                    assert len(subject_labels.index) > 0, f'Skipping empty label file {subject_id}'
                    subject_labels = pd.concat([subject_labels] * len(sample_frame.index))

                    subject_df = pd.concat([sample_frame, subject_labels], axis=1)
                    subject_df.index.name = 'ID'
                    all_subjects.append(subject_df)

                except Exception as e:
                    # Catch empty files
                    print(f"Error {e}, {e.__class__} occurred.")
                    pass

        frame = pd.concat(all_subjects)

        # Do some re-formatting to save some memory
        # TODO: re-format the other columns too, is there a way to automate this?
        # size_before = frame.memory_usage(deep=True).sum()
        frame["nMajor"] = pd.to_numeric(frame["nMajor"], downcast="unsigned")
        frame["nMinor"] = pd.to_numeric(frame["nMinor"], downcast="unsigned")
        frame["GI"] = pd.to_numeric(frame["GI"], downcast="unsigned")
        frame["startpos"] = pd.to_numeric(frame["startpos"], downcast="unsigned")
        frame["endpos"] = pd.to_numeric(frame["endpos"], downcast="unsigned")
        frame["cancer_type"] = frame["cancer_type"].astype("category")
        frame["chr"] = frame["chr"].astype("category")
        frame["WGD"] = frame["WGD"].astype("category")
        # print(f'Achieved compression of {frame.memory_usage(deep=True).sum() / size_before}')

        self.data_frame = frame

        return frame

    def apply_filter(self, cancer_types=None, wgd=None):
        """ Apply feature of interest filters

        @param cancer_types:     list of cancer type flags to keep
        @param wgd:              list of WGD flags to keep
        """
        # Cancer type
        if cancer_types is not None:
            cancer_mask = self.data_frame.cancer_type.isin(cancer_types)
            self.data_frame = self.data_frame[cancer_mask]

        # Whole-genome-doubling
        if wgd is not None:
            wgd_mask = self.data_frame.WGD.isin(wgd)
            self.data_frame = self.data_frame[wgd_mask]

        assert len(self.data_frame.index) > 0, 'There are no samples with these filter criterion'

        return self.data_frame

    def train_test_split(self):
        # Split frame into training, validation, and test
        unique_samples = self.data_frame.index.unique()

        train_labels, test_labels = sk_split(unique_samples, test_size=0.2)
        test_labels, val_labels = sk_split(test_labels, test_size=0.5)
        train_df = self.data_frame[self.data_frame.index.isin(train_labels)]
        test_df = self.data_frame[self.data_frame.index.isin(test_labels)]
        val_df = self.data_frame[self.data_frame.index.isin(val_labels)]
        assert len(train_df.cancer_type.unique()) == len(self.data_frame.cancer_type.unique()),\
            'Check all labels are represented in training set'

        # Random sampler weights
        weight_dict = {}
        ntrain_unique_samples = len(train_df.index.unique())
        for cancer_id, group in train_df.groupby('cancer_type'):
            unique_samples = len(group.index.unique()) / ntrain_unique_samples
            if unique_samples > 0:
                weight_dict[cancer_id] = 1 / unique_samples

        return (train_df, test_df, val_df), weight_dict


class ASCATDataModule(LoadASCAT, pl.LightningDataModule, ABC):
    """

    """

    def __init__(self, batch_size=128, file_path=None, cancer_types=None, wgd=None, custom_edges=None):
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
        self.edges2 = custom_edges
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

        self.train_set = ASCATDataset(self.train_df, self.label_encoder,
                                      weight_dict=self.weight_dict,
                                      custom_edges2seq=self.edges2)
        if self.weight_dict is not None:
            self.train_sampler = WeightedRandomSampler(self.train_set.weights,
                                                       len(self.train_set.weights),
                                                       replacement=True)
            self.train_shuffle = False

        self.test_set = ASCATDataset(self.test_df, self.label_encoder, custom_edges2seq=self.edges2)
        self.validation_set = ASCATDataset(self.val_df, self.label_encoder, custom_edges2seq=self.edges2)

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


class ASCATDataset(Dataset):
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
            chr_length = 100

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

    def default_df2data(self, subject_frame):
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
        self.df2data = custom_df2data if custom_df2data is not None else self.default_df2data
        self.edges2seq = custom_edges2seq if custom_edges2seq is not None else self.default_edges2seq

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
