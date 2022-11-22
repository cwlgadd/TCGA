from sklearn import preprocessing
import TCGA


def example_generator():

    # Create class instance
    ascat = TCGA.data_modules.ascat.LoadASCAT(path='../data_modules/ascat.pkl')

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
