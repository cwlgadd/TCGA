# Methods for loading and parsing the ascat data set into a dataframe
#

from sklearn import preprocessing
from sklearn.model_selection import train_test_split as sk_split
import numpy as np
import pandas as pd
import os
import re
from tqdm import tqdm


class ASCAT:
    """
    Class for loading the ascat data
    """

    def __init__(self, path=None, cancer_types=None, wgd=None):
        self.path = path
        self.filters = {'cancer': cancer_types, 'WGD': wgd}
        self._data_path = os.path.dirname(os.path.abspath(__file__)) + r'/ascat/ReleasedData/TCGA_SNP6_hg19'

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

        # Report any class imbalance
        # df_dict = {'train frame': train_df, 'validation frame': val_df, 'test frame': test_df}
        # for key in df_dict:
        #     assert len(df_dict[key].cancer_type.unique()) == len(cancer_types), f'{key} lost class representation'
        #     for cancer_id, group in df_dict[key].groupby('cancer_type'):
        #         unique_samples = len(group['sample'].unique())
        #         if unique_samples > 0:
        #             print(f'In {key} there are {unique_samples} samples with cancer type {cancer_id}')

        return (train_df, test_df, val_df), weight_dict


def example_generator():

    # Create class instance
    ascat = ASCAT(path=os.path.dirname(os.path.abspath(__file__)) + '/ascat.pkl')

    # Encode remaining cancer type labels, so they can be used by the model later
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit_transform(ascat.data_frame.cancer_type.unique())

    # Split frame into training, validation, and test
    (train_df, test_df, val_df), weight_dict = ascat.train_test_split()
    assert len(train_df.cancer_type.unique()) == len(ascat.data_frame.cancer_type.unique())

    return ascat.data_frame, (train_df, test_df, val_df), weight_dict, label_encoder, ascat


if __name__ == '__main__':

    df, (df_train, df_test, df_val), weight_dict, le, ascat = example_generator()
    print(ascat)

    print(df.head())
    print(df_train.head())
    print(df_test.head())
    print(df_val.head())

    print(f"Weighted random sampler weights: {weight_dict}")
