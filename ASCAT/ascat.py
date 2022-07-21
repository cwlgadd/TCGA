# Methods for loading and parsing the ascat data set into a dataframe
#

import pytorch_lightning as pl
from sklearn import preprocessing
from sklearn.model_selection import train_test_split as sk_split
import numpy as np
import pandas as pd
import os
import re
from tqdm import tqdm

pl.seed_everything(42)

FILE_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = FILE_PATH + r'/ascat/ReleasedData/TCGA_SNP6_hg19'

# Used for transforming true sequences into padded, equal length sequences.
PAD_LENGTH = 1.002
CHROM_LENGTHS = {"1": np.floor(249250621 * PAD_LENGTH),
                 "2": np.floor(243199373 * PAD_LENGTH),
                 "3": np.floor(198022430 * PAD_LENGTH),
                 "4": np.floor(191154276 * PAD_LENGTH),
                 "5": np.floor(180915260 * PAD_LENGTH),
                 "6": np.floor(171115067 * PAD_LENGTH),
                 "7": np.floor(159138663 * PAD_LENGTH),
                 "8": np.floor(146364022 * PAD_LENGTH),
                 "9": np.floor(141213431 * PAD_LENGTH),
                 "10": np.floor(135534747 * PAD_LENGTH),
                 "11": np.floor(135006516 * PAD_LENGTH),
                 "12": np.floor(133851895 * PAD_LENGTH),
                 "13": np.floor(115169878 * PAD_LENGTH),
                 "14": np.floor(107349540 * PAD_LENGTH),
                 "15": np.floor(102531392 * PAD_LENGTH),
                 "16": np.floor(90354753 * PAD_LENGTH),
                 "17": np.floor(81195210 * PAD_LENGTH),
                 "18": np.floor(78077248 * PAD_LENGTH),
                 "19": np.floor(59128983 * PAD_LENGTH),
                 "20": np.floor(63025520 * PAD_LENGTH),
                 "21": np.floor(48129895 * PAD_LENGTH),
                 "22": np.floor(51304566 * PAD_LENGTH),
                 "X": np.floor(155270560 * PAD_LENGTH),
                 "Y": np.floor(59373566 * PAD_LENGTH)
                 }


def _load_ascat(length=500, file_path=None):
    """
    Load ASCAT released count number data.

    :param length: How long the down-sampled vector should be.
    :param file_path: Where to load/save data frame relative to this file.
    :return: A dataframe which will be used in the count number data loader.

    """
    # Load frame from given file_path
    if file_path is not None:
        try:
            return pd.read_pickle(FILE_PATH + file_path)
        except FileNotFoundError:
            print(f"Could not load pickled dataframe from path {FILE_PATH + file_path}, creating new...")

    # Load in the label information -> patient ID, cancer type, WGD, gi
    sample_label_frame = pd.read_csv(DATA_PATH + r'/TCGA.giScores.wgd.txt', delimiter='\t')
    print(f"Value counts for each cancer class: {sample_label_frame.cancer_type.value_counts()}")

    # Load in the count number data
    # Loop over segments files (each file belongs to one patient)
    all_frames = []
    for idx_seg, entry in tqdm(enumerate(os.scandir(DATA_PATH + '/segments/')),
                               desc='Parsing count number data from files',
                               total=12240):

        if (entry.path.endswith(r".segments.txt")) and entry.is_file():
            try:
                # Extract patient identifier from segments file, cross reference against label file, and store labels
                subject_id = re.search(r'segments/(.+?).segments.txt', entry.path).group(1)
                subject_labels = sample_label_frame[sample_label_frame['patient'] == subject_id]
                subject_labels = subject_labels[['cancer_type', 'wgd', 'gi']].to_numpy()[0]

                # Load segment file
                sample_frame = pd.read_csv(entry.path, sep='\t')

                # For each chromosome create windowed CN vectors, normalised so all have 'length' bins.
                # Do not round to integers yet. Here there is a large risk of coarse graining out finer details.
                for chromosome in sample_frame.chr.unique():
                    scale = length / CHROM_LENGTHS[chromosome]
                    sample_frame.loc[sample_frame.chr == chromosome, 'startpos'] *= scale
                    sample_frame.loc[sample_frame.chr == chromosome, 'endpos'] *= scale

                sample_frame.insert(6, 'cancer_type', subject_labels[0])
                sample_frame.insert(7, 'wgd', subject_labels[1])
                sample_frame.insert(8, 'gi', subject_labels[2])

                all_frames.append(sample_frame)

            except Exception as e:
                # Catch empty files
                # print(f"Error {e.__class__} occurred.")
                pass

    frame = pd.concat(all_frames, ignore_index=True)

    # Put into memory efficient formatting
    # size_before = frame.memory_usage(deep=True).sum()
    frame["nmajor"] = pd.to_numeric(frame["nMajor"], downcast="unsigned")
    frame["nminor"] = pd.to_numeric(frame["nMinor"], downcast="unsigned")
    frame["gi"] = pd.to_numeric(frame["gi"], downcast="unsigned")
    frame["startpos"] = pd.to_numeric(frame["startpos"], downcast="unsigned")
    frame["endpos"] = pd.to_numeric(frame["endpos"], downcast="unsigned")
    frame["cancer_type"] = frame["cancer_type"].astype("category")
    frame["chr"] = frame["chr"].astype("category")
    frame["wgd"] = frame["wgd"].astype("category")
    # print(f'Achieved compression of {frame.memory_usage(deep=True).sum() / size_before}')

    # TODO: report how many higher fidelity regions were lost from coarse graining.

    # Save frame to file_path
    if file_path is not None:
        pd.to_pickle(frame, FILE_PATH + file_path)

    return frame


def _setup_ascat(data_frame, cancer_types=None, wgd=None, chromosomes=None):
    """

    :param data_frame:       Frame we want to filter
    :param cancer_types:     list of strings, where each string is the cancer type
    :param wgd:
    :param chromosomes:
    :return:
    """
    # Apply feature of interest filters
    if cancer_types is not None:
        mask = data_frame.cancer_type.isin(cancer_types)
        data_frame = data_frame[mask]
    if wgd is not None:
        data_frame = data_frame[data_frame['wgd'] == wgd]
    if chromosomes is not None:
        raise NotImplementedError('Bugged feature')         # TODO: Not really important to fix atm - just use all
        mask = data_frame.chr.isin(chromosomes)
        data_frame = data_frame[mask]
    assert len(data_frame.index) > 0, 'There are no samples with these filter criterion'

    # Encode remaining cancer type labels, so they can be used by the model later
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit_transform(data_frame.cancer_type.unique())

    # Split frame into training, validation, and test
    unique_samples = data_frame['sample'].unique()
    train_labels, test_labels = sk_split(unique_samples, test_size=0.2)
    test_labels, val_labels = sk_split(test_labels, test_size=0.5)
    train_df = data_frame[data_frame['sample'].isin(train_labels)]
    val_df = data_frame[data_frame['sample'].isin(val_labels)]
    test_df = data_frame[data_frame['sample'].isin(test_labels)]
    assert len(train_df.cancer_type.unique()) == len(data_frame.cancer_type.unique())

    # Random sampler weights
    weight_dict = {}
    for cancer_id, group in train_df.groupby('cancer_type'):
        unique_samples = len(group['sample'].unique()) / len(train_df['sample'].unique())
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

    return data_frame, (train_df, val_df, test_df), weight_dict, label_encoder


def load_ascat(length=500, file_path=None, cancer_types=None, wgd=None, chromosomes=None):

    # Prepare data
    #   load data (parse output files from R code), optionally save when done
    df = _load_ascat(length=length, file_path=file_path)

    # Set up data
    #   splitting into train/test/val sets, create label encoder and training weights
    frame, (df_train, df_val, df_test), train_weights, label_encoder = \
        _setup_ascat(df, cancer_types=cancer_types, wgd=wgd, chromosomes=chromosomes)

    return frame, (df_train, df_val, df_test), train_weights, label_encoder
