# Methods for loading and parsing the ascat data set into a dataframe
#
import copy

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
import random
import pandas as pd
import pyarrow.feather as feather

from TCGA.data_modules.utils.helpers import get_chr_base_pair_lengths as chr_lengths


pl.seed_everything(42)


class LoadASCAT:
    """
    Class for loading and linking the data from various sources.
        Loads Count Number Alteration Data from ASCAT package,
        and then links with survival data (pulled from R::bioconductor and saved to .feather format)
    """

    def __init__(self, path=None, **kwargs):
        self.path = path
        self._data_path = os.path.dirname(os.path.abspath(__file__)) + r'/data/ascat/ReleasedData/TCGA_SNP6_hg19'

        try:
            self.data_frame = self.load(self.path)
        except:
            print(f"Loading from submodule into pandas data frame.")
            self.parse_files()
            if path is not None:
                self.save(self.path)

        # print(self.data_frame.columns)
        self.apply_filter(**kwargs)

        # import matplotlib.pyplot as plt
        # print(self.data_frame.columns)
        # plt.hist(self.data_frame["times"].to_numpy())
        # plt.show()

    def __str__(self):
        s = f'Allele-Specific Copy Number Analysis of Tumors (ASCAT) parser'
        return s

    def load(self, path):
        self.data_frame = pd.read_pickle(path)
        return self.data_frame

    def save(self, path):
        pd.to_pickle(self.data_frame, path)

    def parse_files(self):
        """ Convert the different data sources into a single condensed dataframe so we don't have to load files
            whilst training our PyTorch model.
            
            Sources:
            	summary.ascatv3TCGA.penalty70.hg19.tsv  : summary data for the ASCAT samples
            	segments/(.+?).segments.txt             : the CNA obtained via ASCAT
            	survival_data.feather                   : survival data loaded from R::library(RTCGA) and saved to file

        @return: pandas data frame containing the condensed representation of the ASCAT data (i.e. with start/end pos).
        We do not directly convert to sequence data to save memory - this is done upon loading batches in the datamodule.
        """

        # Load in the label information 
        label_frame = pd.read_csv(self._data_path + r'/summary.ascatv3TCGA.penalty70.hg19.tsv',
                                  delimiter='\t',
                                  index_col='name'
                                  )

        # Load in the survival information
        surv_df = feather.read_feather( os.path.dirname(os.path.abspath(__file__)) +
                                        "/data/survival/survival_data.feather")
        print(surv_df)
                                  
        # Load in the count number data, and link with above frames
        all_subjects = []
        # Loop over segments files (each file belongs to one patient)
        for idx_seg, entry in tqdm(enumerate(os.scandir(self._data_path + '/segments/')),
                                   desc='Parsing count number data from files',
                                   total=len(label_frame.index)):

            # Extract patient identifier from segments file, cross reference against label file, and store labels
            if (entry.path.endswith(r".segments.txt")) and entry.is_file():
                try:
                    # Get patient identifier
                    sample_name = re.search(r'segments/(.+?).segments.txt', entry.path).group(1)      # with -a, -b suffixes

                    # Load segment file corresponding to patient identifier
                    #      index: "sample_name" with suffix
                    #      cols: "chr", "startpos", "endpos", "nMajor", "nMinor"
                    sample_frame = pd.read_csv(entry.path, sep='\t', index_col='sample')
                    assert len(sample_frame.index) > 0, f'Skipping empty segment file {sample_name}'
                    # print(sample_frame)

                    # Find labels corresponding to patient identifier
                    #      index: "sample_name" with suffix        (repeated)
                    subject_labels = label_frame.loc[[sample_name]]
                    assert len(subject_labels.index) > 0, f'Skipping empty label file {sample_name}'
                    subject_labels = pd.concat([subject_labels] * len(sample_frame.index))
                    subject_labels.index.name = 'sample'
                    # print(subject_labels)

                    # Find  survival data corresponding to patient identifier. In case of duplication or double entry, take last.
                    #      index: "sample_name" with suffix        (repeated)
                    subject_surv = surv_df.loc[surv_df['bcr_patient_barcode'] == subject_labels['patient'][0]].drop_duplicates()
                    assert subject_surv["times"].size > 0, f"skipping missing survival entry {sample_name}"                  # and (subject_surv["patient.vital_status"].size > 0
                    assert subject_surv["times"].size == 1, f"skipping doubled (non-duplicated) survival entry {sample_name}"       #  and (subject_surv["patient.vital_status"].size == 1)
                    subject_surv["sample"] = sample_name
                    subject_surv.set_index("sample", inplace=True)
                    subject_surv = pd.concat([subject_surv] * len(sample_frame.index))
                    # print(subject_surv)
                    
                    subject_df = pd.concat([sample_frame, subject_labels, subject_surv], axis=1)
                    # print(subject_df)
                    all_subjects.append(subject_df)

                except Exception as e:
                    # Catch empty files
                    print(f"Error: {e}")  #, {e.__class__} occurred.")
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

    def apply_filter(self, cancer_types=None, wgd=None, cna_clip=None):
        """ Apply feature of interest filters

        @param cancer_types:     list of cancer type flags to keep
        @param wgd:              list of WGD flags to keep
        @param cna_clip:         number of deviations from the mean to clip
        """

        # Cancer type
        if cancer_types is not None:
            cancer_mask = self.data_frame.cancer_type.isin(cancer_types)
            self.data_frame = self.data_frame[cancer_mask]

        # Whole-genome-doubling
        if wgd is not None:
            wgd_mask = self.data_frame.WGD.isin(wgd)
            self.data_frame = self.data_frame[wgd_mask]

        # Clip outlier CNA
        if cna_clip is not None:
            # print(self.data_frame['nMajor'].std() * cna_clip)
            # print(self.data_frame['nMinor'].std() * cna_clip)
            self.data_frame['nMajor'] = self.data_frame['nMajor'].clip(upper=self.data_frame['nMajor'].std() * cna_clip)
            self.data_frame['nMinor'] = self.data_frame['nMinor'].clip(upper=self.data_frame['nMinor'].std() * cna_clip)

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


class ASCATDataModule(pl.LightningDataModule):
    """
    Cancer types:

    """
    train_set, test_set, validation_set = None, None, None
    train_sampler, train_shuffle = None, True
    label_encoder = None

    @property
    def data_frame(self):
        return self.ascat.data_frame

    @property
    def labels(self):
        return self.label_encoder.classes_

    def __init__(self,
                 batch_size=128,
                 file_path=None,
                 chrom_as_channels=True,
                 custom_edges=None,
                 scaler=None,
                 **kwargs):
        """

        @param batch_size:
        @param file_path:
            When first loading from file, individual patient files are condensed to a single file.
             This is the location of the pickle file data is saved to after this pre-processing step.
        @param chrom_as_channels:
            Whether we treat each chromosome as separate channel (True), or stack them along locus position (False)
        @param cancer_types:
            ['ACC' 'BLCA' 'BRCA' 'CESC' 'CHOL' 'COAD' 'DLBC' 'ESCA' 'GBM' 'HNSC'
             'KICH' 'KIRC' 'KIRP' 'LAML' 'LGG' 'LIHC' 'LUAD' 'LUSC' 'MESO' 'OV' 'PAAD'
             'PCPG' 'PRAD' 'READ' 'SARC' 'SKCM' 'STAD' 'TGCT' 'THCA' 'THYM' 'UCEC'
             'UCS' 'UVM']
        @param wgd:
            Filter based on whether WGD has occurred (based on average copy number)
        @param custom_edges:
            Optional custom function wrapper for dataloader, for converting start_pos + end_pos CNA to signals
        @param sampler:
            Boolean flag whether we use a weighted random sampler
        """
        if file_path is None:
            file_path = os.path.dirname(os.path.abspath(__file__)) + '/data/ascat.pkl'

        super().__init__()

        self.batch_size = batch_size
        self.file_path = file_path
        self.chrom_as_channels = chrom_as_channels
        self.edges2 = custom_edges
        self.scaler = scaler

        self.ascat = LoadASCAT(path=self.file_path, **kwargs)

        self.setup()
        self.W = self.train_set.chr_length if self.chrom_as_channels else self.train_set.chr_length * 23
        self.C = 2*23 if self.chrom_as_channels else 2

    def __str__(self):
        s = "\nASCATDataModule"
        for df, data_set in zip([self.train_df, self.val_df, self.test_df], ["Train", "Validation", "Test"]):
            df = df.groupby('sample').first()
            s += f"\n\t {data_set}"
            for i, j in zip(df['cancer_type'].value_counts().keys(), df['cancer_type'].value_counts().values):
                if j > 0:
                    s += f"\n\t\t {i.ljust(8)}: {j}"
        return s

    def prepare_data(self):
        pass

    def setup(self, stage=None):

        # Create label encoder
        self.label_encoder = preprocessing.LabelEncoder()
        self.label_encoder.fit_transform(self.data_frame.cancer_type.unique())

        # Get train-test-val splits (this is in a dense form of start/end positions and counts)
        (self.train_df, self.val_df, self.test_df), _ = self.ascat.train_test_split()

        # Create training set so we can iterate over it to fit scaler. It must be done this way as we do not store the
        #   data in its full form to initialise the Dataset with a preprocessing transformation
        self.train_set = ASCATDataset(self.train_df,
                                      self.label_encoder,
                                      chrom_as_channels=self.chrom_as_channels,
                                      custom_edges2seq=self.edges2)

        # Pre-processing - TODO: this is really inefficient pre-processing, add chunking
        if self.scaler is not None:
            min_max_keys = ["survival_time"]
            other_keys = ["covariates", "CNA"]
            batch_keys = [_key for _key in next(iter(self.train_set)).keys() if _key in min_max_keys + other_keys]
            stacked_batches = dict(zip(batch_keys, [[] for _key in batch_keys if _key in min_max_keys + other_keys]))
            # Stack batches
            for batch in self.train_set:
                for _key in batch_keys:
                    stacked_batches[_key].append(batch[_key])
            for _key in batch_keys:
                stacked_batches[_key] = torch.stack(stacked_batches[_key], dim=0)
            # Train scalers
            scalers = dict(zip(batch_keys, [copy.copy(self.scaler) if _key in other_keys
                                            else preprocessing.MinMaxScaler()
                                            for _key in batch_keys ]))
            for _key in batch_keys:
                scalers[_key].fit(stacked_batches[_key].reshape((len(self.train_set), -1)))
            self.train_set.scaler_dict = scalers

        self.test_set = ASCATDataset(self.test_df,
                                     self.label_encoder,
                                     chrom_as_channels=self.chrom_as_channels,
                                     custom_edges2seq=self.edges2,
                                     scaler_dict=self.train_set.scaler_dict)
        self.validation_set = ASCATDataset(self.val_df,
                                           self.label_encoder,
                                           chrom_as_channels=self.chrom_as_channels,
                                           custom_edges2seq=self.edges2,
                                           scaler_dict=self.train_set.scaler_dict)

    def train_dataloader(self):
        return DataLoader(
            sampler=self.train_sampler,
            dataset=self.train_set,
            batch_size=self.batch_size,
            num_workers=np.min((8,os.cpu_count())),
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.validation_set,
            batch_size=self.batch_size,
            num_workers=np.min((8,os.cpu_count())),
            shuffle=True
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_set,
            batch_size=self.batch_size,
            num_workers=np.min((8,os.cpu_count())),
            shuffle=False,
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
            chr_length = self.chr_length

            CNA_sequence = torch.ones((2, 23, chr_length))
            for row in subject_edge_info.iterrows():
                chrom = 23 if row[1]['chr'] in ['X', 'Y'] else int(row[1]['chr'])

                start_pos = int(np.floor(row[1]['startpos'] / true_chr_lengths[row[1]['chr']] * chr_length))
                end_pos = int(np.floor(row[1]['endpos'] / true_chr_lengths[row[1]['chr']] * chr_length))

                # Major strand
                CNA_sequence[0, chrom-1, start_pos:end_pos] = row[1]['nMajor'] # np.log1p()
                # Minor strand
                CNA_sequence[1, chrom-1, start_pos:end_pos] = row[1]['nMinor'] # np.log1p()

        else:
            # TODO: Above assumes each chromosome has equal length - implement alternative with zero-padding
            raise NotImplementedError

        if self.chrom_as_channels:
            return CNA_sequence.reshape((-1, chr_length))
        else:
            return CNA_sequence.reshape((2, -1))

    def default_df2data(self, subject_frame):
        """
        :return:  x shape (seq_length, num_channels, num_sequences)
        """
        subject_frame.reset_index(drop=True, inplace=True)
        # print(subject_frame.columns)

        # Get the columns relevant for the CNA
        subject_edge_info = subject_frame[['startpos', 'endpos', 'nMajor', 'nMinor', 'chr']]
        count_numbers = self.edges2seq(subject_edge_info)

        # Get ASCAT labels
        cancer_name = subject_frame["cancer_type"][0]
        label = list(self.label_encoder.classes_).index(cancer_name)

        # Get Surv labels
        surv_time = subject_frame["times"][0]
        surv_status = subject_frame["patient.vital_status"][0]
        # Baseline covariates
        gender = torch.tensor(1 if subject_frame["patient.gender"][0] == "male" else 0)
        surv_age = torch.tensor(subject_frame["patient.days_since_birth"][0])
        baseline_covariates = torch.stack(tensors=(surv_age, gender), dim=0)

        return {'covariates': baseline_covariates.float(),
                'CNA': count_numbers.float(),
                'label': torch.tensor(label).float(),
                'survival_time': torch.tensor(surv_time).float(),
                'survival_status': torch.tensor(surv_status).float(),
                }

    def __init__(self, data: pd.DataFrame, label_encoder, weight_dict: dict = None, chrom_as_channels=True,
                 custom_df2data=None, custom_edges2seq=None, scaler_dict=None):
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
        self.chrom_as_channels = chrom_as_channels
        self.scaler_dict = scaler_dict
        
        # custom wrappers
        self.df2data = custom_df2data if custom_df2data is not None else self.default_df2data
        self.edges2seq = custom_edges2seq if custom_edges2seq is not None else self.default_edges2seq

        _tmp = self.data_frame.groupby('sample').first()
        self.IDs = [row[0] for row in _tmp.iterrows()]
        if weight_dict is not None:
            self.weights = [weight_dict[row[1].cancer_type] for row in _tmp.iterrows()]

    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        subject_frame = self.data_frame.loc[[self.IDs[idx]]]
        s_frame = self.df2data(subject_frame)

        if self.scaler_dict is not None:
            for _key in self.scaler_dict.keys():
                scaled_ = self.scaler_dict[_key].transform(s_frame[_key].numpy().reshape((1, -1)))
                s_frame[_key] = torch.from_numpy(scaled_.reshape(s_frame[_key].shape))

        return s_frame


if __name__ == "__main__":

    dm = ASCATDataModule(batch_size=256,
                         cancer_types=['THCA', 'BRCA', 'OV', 'GBM', 'HNSC'],
                         cna_clip=3,
                         chrom_as_channels=True,
                         scaler=preprocessing.MinMaxScaler())

    for loader in [dm.train_dataloader(), dm.val_dataloader(), dm.test_dataloader()]:
        for batch in loader:
            for key in batch.keys():
                print(f"{key}:".ljust(30) +
                      f"{batch[key].shape}".ljust(30) +
                      f"({torch.min(batch[key])}, {torch.max(batch[key])})")
            break

    print(torch.mean(batch["covariates"][:, 0]))