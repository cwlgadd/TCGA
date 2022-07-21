# Methods for loading and parsing the simulated version of the ascat dataset into a dataframe
#

import copy
from sklearn import preprocessing
from sklearn.model_selection import train_test_split as sk_split
import numpy as np
import pandas as pd

# import os
# FILE_PATH = os.path.dirname(os.path.realpath(__file__))


class SimulateMarkov:
    """
    Create or load simulated version of ASCAT count number data.
    """

    @property
    def create_insertion_kernel(self):
        """ Sample beginning pos and length of a transition, of all +1
       """
        _max_width = np.min((20, np.floor(2 * self.length / 3)))

        delta_state = np.zeros(self.length)
        start = np.random.randint(0, self.length - (_max_width - 1))  # Sample (from 0 to length-5)
        width = np.random.randint(1, (_max_width + 1))  # Number of elements that change (from 1 to 5)
        delta_state[start:start + width] += 1
        return delta_state

    @property
    def t(self):
        return self.trajectories.shape[1]

    @property
    def noise_kernel(self):
        """ With some probability we perform a transition that is purely random"""
        #TODO: not implemented
        return np.random.choice([0, 1], size=(self.length,), p=[4. / 5, 1. / 5])

    def __init__(self, classes=2, length=100, n=50000, n_kernels_per=30, n_kernels_shared=20, path=None):
        assert n_kernels_shared < n_kernels_per
        self.classes = classes
        self.length = length
        self.n = n
        self.n_kernels = n_kernels_per
        self.n_kernels = n_kernels_shared
        self.path = path

        _shared_kernels = [self.create_insertion_kernel for _ in range(n_kernels_shared)]
        self.kernels = [
            [self.create_insertion_kernel for _ in range(n_kernels_per - n_kernels_shared)] + _shared_kernels
            for _ in range(classes)
        ]

        self.trajectories = np.ones((n, 1, length))     # N x T x D
        self.labels = np.random.choice([i for i in range(classes)],
                                       size=n,
                                       p=[1./classes for _ in range(classes)]
                                       )

    def __str__(self):
        s = ""
        for idx_c, c in enumerate(self.kernels):
            s += f"\nClass {idx_c}"
            s += f"\n{np.vstack(c)}"
        return s

    def __call__(self, steps=1):
        """ Sample through the Markov Process
        """
        s_t = np.zeros((self.n, steps + 1, self.length))
        s_t[:, 0, :] = self.trajectories[:, -1, :]

        # TODO: vectorise
        for step in range(steps):
            delta = np.zeros_like(self.trajectories[:, -1, :])
            for label in range(self.classes):
                next_kernel = \
                    np.random.choice([i for i in range(len(self.kernels[label]))], size=s_t.shape[0],
                                     p=[1. / len(self.kernels[label]) for _ in range(len(self.kernels[label]))]
                                     )
                for k_ind, kernel in enumerate(self.kernels[label]):
                    delta[next_kernel == k_ind, :] = np.tile(kernel, (sum(next_kernel == k_ind), 1))
            s_t[:, step + 1, :] = s_t[:, step, :] + delta

        self.trajectories = np.concatenate((self.trajectories, s_t[:, 1:, :]), axis=1)
        return self.trajectories

    def make_data_frame(self, load=False):
        """ Store final state in dataframe """

        # Load frame from given file_path
        if self.path is not None and load is True:
            try:
                return pd.read_pickle(self.path)
            except FileNotFoundError:
                print(f"Could not load pickled dataframe from path {self.path}, creating new...")

        # Put into dataframe
        frame = pd.DataFrame(self.trajectories[:, -1, :], columns=[f'gene{i}' for i in range(self.length)])
        frame['labels'] = self.labels
        frame["labels"] = frame["labels"].astype("category")

        # Save frame to file_path
        if self.path is not None:
            pd.to_pickle(frame, self.path)

        return frame


def example():

    # Create class instance
    data_generator = SimulateMarkov()
    print(data_generator)

    # Run markov process forward
    chain = data_generator(steps=int(100))
    print(f"Now we have a trajectory of shape {chain.shape}")

    df = data_generator.make_data_frame()
    print(df)

    # Encode remaining cancer type labels, so they can be used by the model later
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit_transform(df.labels.unique())

    # Split frame into training, validation, and test
    train_df, test_df = sk_split(df, test_size=0.2)
    test_df, val_df = sk_split(test_df, test_size=0.2)
    assert len(train_df.labels.unique()) == len(df.labels.unique())

    weight_dict = None

    return df, (train_df, val_df, test_df), weight_dict, label_encoder, data_generator


if __name__ == '__main__':

    frame, (df_train, df_val, df_test), _, le, _ = example()
    print(frame.head())
