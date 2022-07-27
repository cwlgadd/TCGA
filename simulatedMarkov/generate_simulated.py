# Methods for loading and parsing the simulated version of the ascat dataset into a dataframe
#
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os



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
        # TODO: not implemented
        return np.random.choice([0, 1], size=(self.length,), p=[4. / 5, 1. / 5])

    def __init__(self, classes=2, channels=2, length=200, n=1000, n_kernels_per=2, n_kernels_shared=1,
                 path=None):
        """

        """
        assert n_kernels_shared < n_kernels_per
        self.classes = classes
        self.strands = 2
        self.channels = channels
        self.length = length
        self.n = n
        self.n_kernels = n_kernels_per
        self.n_kernels = n_kernels_shared
        if path is None:
            path = os.path.dirname(os.path.abspath(__file__))
        self.path = path

        # Create the kernels that are shared between each class
        _shared_kernels = [self.create_insertion_kernel for _ in range(n_kernels_shared)]
        # Create the kernels that are exclusive to each class (but currently shared across strands + chromosomes)
        self.kernels = [
            [self.create_insertion_kernel for _ in range(n_kernels_per - n_kernels_shared)] + _shared_kernels
            for _ in range(classes)
        ]

        self.trajectories = np.ones((n, 1, self.strands, channels, length))     # N x T x D
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
        s_t = np.zeros((self.n, steps + 1, self.strands, self.channels, self.length))
        s_t[:, 0, :, :, :] = self.trajectories[:, -1, :, :, :]

        # TODO: vectorise
        for step in range(steps):
            print(f"step {step}")
            delta = np.zeros_like(self.trajectories[:, -1, :, :, :])

            for n in range(self.n):
                # print(f"sample {n}")
                for strand in range(self.strands):
                    for channel in range(self.channels):
                        kernel_idx = np.random.choice([i for i in range(len(self.kernels[self.labels[n]]))])
                        kernel = self.kernels[self.labels[n]][kernel_idx]
                        delta[n, strand, channel, :] += kernel
            s_t[:, step + 1, :, :, :] = s_t[:, step, :, :, :] + delta

            # delta = np.zeros_like(self.trajectories[:, -1, :, :, :])
            # for label in range(self.classes):
            #     next_kernel = \
            #         np.random.choice([i for i in range(len(self.kernels[label]))],
            #                          size=(self.n, self.strands, self.channels),
            #                          p=[1. / len(self.kernels[label]) for _ in range(len(self.kernels[label]))]
            #                          )
            #     for k_ind, kernel in enumerate(self.kernels[label]):
            #         delta[self.labels == label and next_kernel[0] == k_ind,
            #               next_kernel[1] == k_ind,
            #               next_kernel[2] == k_ind, :] = 1 #np.tile(kernel, (sum(next_kernel == k_ind), 1))
            # s_t[:, step + 1, :, :, :] = s_t[:, step, :, :, :] + delta

        self.trajectories = np.concatenate((self.trajectories, s_t[:, 1:, :, :, :]), axis=1)
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
        columns = [f'strand {strand}' for strand in range(self.strands)]
        a = []
        for n in range(self.n):
            A = [[self.trajectories[n, -1, strand, c, :] for c in range(self.channels)] for strand in range(self.strands)]
            a.append(A)

        frame = pd.DataFrame(a, columns=columns)
        # print(frame.iloc[0]['nMajor'][0])

        frame['labels'] = self.labels
        frame["labels"] = frame["labels"].astype("category")

        # Save frame to file_path
        if self.path is not None:
            pd.to_pickle(frame, self.path + '/simulatedMarkov.pkl')

        return frame


def example_generator():

    # Create class instance
    data_generator = SimulateMarkov(n=10000, classes=2, channels=10, n_kernels_per=3, n_kernels_shared=1)
    print(data_generator)

    # Run markov process forward
    chain = data_generator(steps=int(2))
    print(f"Now we have a trajectory of shape {chain.shape},"
          f" with dimensions (samples x time steps x strands x channels (chr) x sequence length")

    df = data_generator.make_data_frame()
    print(df)

    # Encode remaining cancer type labels, so they can be used by the model later
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit_transform(df.labels.unique())

    # Split frame into training, validation, and test
    train_df, test_df = train_test_split(df, test_size=0.2)
    test_df, val_df = train_test_split(test_df, test_size=0.2)
    assert len(train_df.labels.unique()) == len(df.labels.unique())

    weight_dict = None

    return df, (train_df, val_df, test_df), weight_dict, label_encoder, data_generator


if __name__ == '__main__':

    df, (df_train, df_val, df_test), _, le, _ = example_generator()
