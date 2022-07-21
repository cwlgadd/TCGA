# Methods for loading and parsing the simulated version of the ascat dataset into a dataframe
#

import copy
import pytorch_lightning as pl
from sklearn import preprocessing
from sklearn.model_selection import train_test_split as sk_split
import numpy as np
import pandas as pd
import os

pl.seed_everything(42)

FILE_PATH = os.path.dirname(os.path.realpath(__file__))


def _load_mdp(length=20, steps=5, n=1000, file_path=None):
    """
    Create or load simulated version of ASCAT count number data.

    :param length: How long the down-sampled vector should be.
    :param num_kernels: The number of transition kernels to use
    :param file_path: Where to load/save data frame relative to this file.
    :return: A dataframe which will be used in the count number data loader.

    """

    def sample_transition1(length_=10, max_width_=5):
        """ This transition samples beginning and length of a transition, of all +1
        """
        assert length_ >= 10, 'transition1 moves written to require length at least 10'
        delta_state = np.zeros(length_)
        start = np.random.randint(0, length_-(max_width_-1))              # Sample (from 0 to length-5)
        width = np.random.randint(1, (max_width_+1))                      # Number of elements that change (from 1 to 5)
        delta_state[start:start+width] += 1
        return delta_state

    def sample_mdp(kernel_, steps_=10, n_=1000):
        """ Given a kernel, sample through the Markov Decision Process
        """
        basis_length = len(kernel_[0][1])

        def noise(probability):
            if np.random.uniform(0, 1) < probability:
                return np.random.choice([0, 1], size=(basis_length,), p=[4./5, 1./5])
            else:
                return 0

        samples = np.zeros((n_, basis_length))
        chain = [samples]
        for step in range(steps_):
            samples = copy.deepcopy(samples)
            for i in range(n_):
                idx_transition = np.random.choice(len(kernel_), p=[transition[0] for transition in kernel_])
                samples[i, :] += kernel_[idx_transition][1] + noise(0.001)
            chain.append(samples)
        return chain

    # Load frame from given file_path
    if file_path is not None:
        try:
            return pd.read_pickle(FILE_PATH + file_path)
        except FileNotFoundError:
            print(f"Could not load pickled dataframe from path {FILE_PATH + file_path}, creating new...")

    # Build custom transition kernel.
    classes = 2
    num_shared, num_each, max_width = 0, 2, 5

    shared_basis = [sample_transition1(length, max_width_=max_width) for _ in range(num_shared)]

    def make_kernel(_num_shared, _num_each, _max_width):
        return list(zip([1. / _num_each for _ in range(_num_each)],
                        shared_basis + [sample_transition1(length, max_width_=_max_width)
                                        for _ in range(_num_each - _num_shared)]
                        ))
    kernels = [make_kernel(num_shared, num_each, max_width) for _ in range(classes)]

    # Progressive kernel distinction test
    #kernels = [make_kernel(num_shared, num_each, max_width)]
    #for c in range(1, classes):
    #    num_each += 1
    #    next_kernel = list(zip([1./num_each for _ in range(num_each)],
    #                           [kernels[-1][i][1] for i in range(len(kernels[-1]))] +
    #                           [sample_transition1(length, max_width)]
    #                           ))
    #    kernels.append(next_kernel)

    # Check and report
    for idx_kern, kernel in enumerate(kernels):
        # Report each transition
        print(f'Kernel{idx_kern} bases: ')
        for idx_basis, transition in enumerate(kernel):
            print(f"Basis{idx_basis} with probability {transition[0]}: {transition[1]}")

        assert np.isclose(np.sum([transition[0] for transition in kernel]), 1), f"Probabilities should total 1. {kernel}"
        #assert len(np.unique([transition[1] for transition in kernel], axis=0)) == len(kernel),\
        #    f"Replicated transition. {len(np.unique([transition[1] for transition in kernel], axis=0))} != {len(kernel)}"

    # Sample MDP, reshape, and get labels
    markov_chains = [sample_mdp(kernel, steps_=steps, n_=n) for idx, kernel in enumerate(kernels)]
    markov_chains = np.concatenate(markov_chains, axis=1)

    labels_sep = [[f'Kernel{idx}' for _ in range(n)] for idx in range(len(kernels))]
    labels = [y for x in labels_sep for y in x]

    # View first sample
    # import matplotlib.pyplot as plt
    # for i in range(steps):
    #     plt.scatter(np.linspace(0, length, length), markov_chain[i, 0, :])
    #     plt.show()

    # Format into string to make dataframe more interpretable
    samples = []
    for i in range(markov_chains[0].shape[0]):
        step_ = []
        for step in range(len(markov_chains)):
            string = ""
            for element in markov_chains[step][i, :]:
                string += str(int(element)) + ","
            string = string[:-1]
            step_.append(string)
        samples.append(step_)

    # Put into dataframe
    frame = pd.DataFrame(samples, columns=[f't{i}' for i in range(steps+1)])
    frame['labels'] = labels
    frame["labels"] = frame["labels"].astype("category")

    # Save frame to file_path
    if file_path is not None:
        pd.to_pickle(frame, FILE_PATH + file_path)

    return frame, kernels


def _setup_mdp(data_frame):
    """

    :param data_frame:       Frame we want to filter, split, and encode
    :return:
    """
    # Apply feature of interest filters (kept for completeness)

    # Encode remaining cancer type labels, so they can be used by the model later
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit_transform(data_frame.labels.unique())

    # Split frame into training, validation, and test
    train_df, test_df = sk_split(data_frame, test_size=0.2)
    test_df, val_df = sk_split(test_df, test_size=0.2)
    assert len(train_df.labels.unique()) == len(data_frame.labels.unique())

    # Random sampler weights (kept for completeness - this doesn't in practice do anything)
    #weight_dict = {}
    #for label_id, group in train_df.groupby('labels'):
    #    unique_samples = 1  # len(group['sample'].unique()) / len(train_df['sample'].unique())
    #    if unique_samples > 0:
    #        weight_dict[label_id] = 1 / unique_samples
    weight_dict = None

    return data_frame, (train_df, val_df, test_df), weight_dict, label_encoder


def load_mdp(length=25, steps=5, n=10000, file_path=None):
    df, kernels = _load_mdp(length=length, steps=steps, n=n, file_path=file_path)
    data_frame, (train_df, val_df, test_df), weight_dict, label_encoder = _setup_mdp(df)
    return data_frame, (train_df, val_df, test_df), weight_dict, label_encoder, kernels


if __name__ == '__main__':

    frame, (df_train, df_val, df_test), train_weights, label_encoder, kernels = load_mdp()
    print(frame.head())
