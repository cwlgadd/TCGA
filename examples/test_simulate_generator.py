import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split as sk_split
import TCGA


def debug_generator():

    # Create class instance
    data_simulator = TCGA.data_modules.simulated.SimulateMarkov(length=20,
                                                                n=10000,
                                                                n_class_bases=4,
                                                                n_bases_shared=0,
                                                                init_steps=1)

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
