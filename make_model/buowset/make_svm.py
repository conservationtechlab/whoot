"""Make SVM with buowset data

This script can be run to create an SVM from buowset data.

If you already have an embeddings dataframe where the key is the filename
and the value is the embedding for that file:

Usage: python3 make_svm.py -embed_df /path/to/premade/df.csv

If you would like to save our your resulting model file, add
    -model_file /path/to/save/model.pkl

"""
import argparse
import pickle
import sys
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import numpy as np
from embed_to_df_birdnet import obtain_birdnet_embeddings


# folds to use for training
TRAINING_FOLDS = [0,1,2,3]
# fold to use for testing
TESTING_FOLDS = [4]
# no buow is 5th class, to be marked as 0 for a binary svm, nums not listed
# will be marked as 1
CLASS_0 = 5


def get_binary_classes(df):

    df['binary_label'] = (df['label'] != CLASS_0).astype(int)
    return df


def make_x_and_y(embed_df):
    """Create train and test split based on existing folds

    Default, this will create an 80% train 20% test split with
    the 5th fold data as the test data, with the buow segments
    as 1 and the no_buow segments as 0.

    Args:
        embed_df (pd.Dataframe): Dictionary inside a dataframe with the
                                 filename as the key and the list of
                                 floats (embeddings) as the value.

    Returns:
        x_train (np.array): all of the burrowing owl detection
                            embeddings to train

        y_train (np.array): all of the no_buow detection embeddings to train

        x_test (np.array): all of the burrowing owl detection
                           embeddings to test

        y_test (np.array): all of the no_buow detection embeddings to test
    """
    train_df = embed_df[embed_df['fold'].isin(TRAINING_FOLDS)]
    test_df = embed_df[embed_df['fold'].isin(TESTING_FOLDS)]

    embedding_cols = embed_df.select_dtypes(include='float64').columns.tolist()

    x_train = train_df[embedding_cols].values
    y_train = train_df['binary_label'].values
    x_test = test_df[embedding_cols].values
    y_test = test_df['binary_label'].values

    return x_train, y_train, x_test, y_test


def make_svm(embeddings_df):
    """Obtain embeddings, train test split, and create an SVM

    Args:
        embeddings_df (str): the path to your embeddings folds/files.

    Returns:
        svm (model): Support vector machine model.
    """
    x_train, y_train, x_test, y_test = make_x_and_y(embeddings_df)
    print("beginning model training")
    svm = SVC(class_weight='balanced', probability=True)
    svm.fit(x_train, y_train)

    y_pred_default = svm.predict(x_test)

    print("Classification report with default threshold:")
    print(classification_report(y_test, y_pred_default))

    return svm


def save_out_model(svm, model_file):
    """Saves model as a pkl.

    If you'd like to save the model to use, you can optionally
    provide a -model_file arg string. If you are just wanting
    to see how the model metrics are or testing code, you
    may not want to save the model each time.

    Args:
        svm (model): The support vector machine created.

        model_file (str): Path to where the model will be saved .pkl.
    """
    with open(model_file, 'wb') as file:
        pickle.dump(svm, file)

    print(f"Saved model path: {model_file}")


def main(embed_df, model_file):
    """Main script to run

    Args:
        embed_df (str): If you have a premade dataframe with the keys
                             as the filename and the values as the embedding
                             list, use this argument and leave embeds and
                             source empty.

        model_file (str): Path to desired model output file, must be a .pkl.
    """
    dataset = pd.read_pickle(embed_df)

    dataset = get_binary_classes(dataset)

    if model_file is None:
        make_svm(dataset)
    else:
        svm = make_svm(dataset)
        save_out_model(svm, model_file)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(
        description='Input Directory Path'
    )
    PARSER.add_argument('-embed_df', type=str, default=None,
                        help='Path to your premade embeddings dataframe.')
    PARSER.add_argument('-model_file', type=str, default=None,
                        help='File name and location of saved model.pkl.')
    ARGS = PARSER.parse_args()
    main(ARGS.embed_df, ARGS.model_file)
