# Methods for loading and parsing the simulated version of the ascat dataset into a dataframe.
from sklearn import preprocessing
from sklearn.model_selection import train_test_split as sk_split
import numpy as np
import pandas as pd


class SimulateMarkov:
    """
    Create or load simulated version of ASCAT count number data.
    """

    def make_basis(self):
        """ Sample beginning pos and length of a transition, of all +1
       """
        _max_width = np.min((20, np.floor(2 * self.length / 3)))

        delta_state = np.zeros(self.length)
        start = np.random.randint(0, self.length - (_max_width - 1))  # Sample (from 0 to length-5)
        width = np.random.randint(1, (_max_width + 1))  # Number of elements that change (up to _max_width)
        delta_state[start:start + width] += 1
        return delta_state

    @property
    def last_state(self):
        return self.trajectories[:, -1, :]

    @property
    def frame(self):
        # Creates a new frame every call, #TODO
        # This is what is read into the loader
        d = {'features': list(self.last_state), 'labels': self.labels}
        return pd.DataFrame(data=d)

    def __init__(self, classes=2, length=100, n=50000, n_class_bases=30, n_bases_shared=20, path=None, init_steps=1):
        assert n_bases_shared < n_class_bases
        self.classes = classes
        self.length = length
        self.n = n
        self.n_kernels = n_class_bases
        self.n_kernels = n_bases_shared
        self.path = path

        # Create the shared kernels/bases
        _shared_bases = [self.make_basis() for _ in range(n_bases_shared)]
        # Create the remaining, independent, kernels/bases of each class
        self.bases = [
            [self.make_basis() for _ in range(n_class_bases - n_bases_shared)] + _shared_bases
            for _ in range(classes)
        ]

        # Begin all samples at the 1 count at every locus
        self.trajectories = np.ones((n, 1, length))     # N x T x D
        # And randomly assign each sample to a different class with equal probability
        self.labels = np.random.choice([i for i in range(classes)], size=n, p=[1./classes for _ in range(classes)])

        if init_steps > 0:
            self(init_steps)

    def __str__(self):
        s = "SimulateMarkov class summary\n==========================="
        for idx_c, c in enumerate(self.bases):
            s += f"\nClass {idx_c} has bases:"
            s += f"\n{np.vstack(c)}"

        combinations = np.unique(self.last_state, axis=0)
        s += f"\n ... giving (end of trajectory, steps={self.trajectories.shape[1]-1}) " \
             f"{combinations.shape[0]} combinations:\n{combinations}"
        return s

    def __call__(self, steps=1):
        """ Sample through the Markov Process
        """
        s_t = np.zeros((self.n, steps + 1, self.length))
        s_t[:, 0, :] = self.last_state

        for step in range(steps):
            delta = np.zeros_like(self.last_state)
            for label in range(self.classes):
                n_basis_in_class = len(self.bases[label])
                idx_next_basis = np.random.choice([i for i in range(n_basis_in_class)], size=sum(self.labels == label),
                                                  p=[1. / n_basis_in_class for _ in range(n_basis_in_class)]
                                                  )
                delta_class = np.vstack([self.bases[label][idx] for idx in idx_next_basis])
                delta[self.labels == label] = delta_class
            s_t[:, step + 1, :] = s_t[:, step, :] + delta

        self.trajectories = np.concatenate((self.trajectories, s_t[:, 1:, :]), axis=1)

        return self.trajectories


def debug_generator():
    # TODO: turn into unit test

    # Create class instance
    data_simulator = SimulateMarkov(length=20, n=10000, n_class_bases=4, n_bases_shared=0, init_steps=1)
    print(data_simulator)

    # Encode labels
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit_transform(np.unique(data_simulator.labels, axis=0))

    # Split frame into training, validation, and test
    train_df, test_df = sk_split(data_simulator.frame, test_size=0.2)
    test_df, val_df = sk_split(test_df, test_size=0.2)
    assert len(train_df.labels.unique()) == len(data_simulator.frame.labels.unique())

    return data_simulator.frame, (train_df, val_df, test_df), label_encoder, data_simulator


if __name__ == '__main__':

    df, (df_train, df_val, df_test), le, ds = debug_generator()
