"""Make SVM with buowset data

This script can be run to create an SVM from buowset data.

If you already have an embeddings dataframe where the key is the filename
and the value is the embedding for that file:

Usage: python3 make_svm.py -df /path/to/premade/df.csv
    -meta /path/to/metadata.csv

If you would like the embeddings dataframe to be created for you and passed
to the script to make the svm:

Usage: python3 make_svm.py -meta /path/to/fold/metadata.csv
    -embeds /path/to/embedding/directory/or/file -source birdnet (or perch)

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
from embeddings_to_df import obtain_birdnet_embeddings
from embeddings_to_df import obtain_perch_embeddings


def make_x_and_y(data, embed_df):
    """Create train and test split based on existing folds

    Default, this will create an 80% train 20% test split with
    the 5th fold data as the test data, with the buow segments
    as 1 and the no_buow segments as 0.

    Args:
        data (pd.Dataframe): Metadata file for buowset with fold info and
                             labels as ints.

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
    x_train = []  # 4 of the folds
    y_train = []  # 4 of the folds
    x_test = []  # the small one 20%, 1 folds worth
    y_test = []  # the small one 20%, 1 folds worth
    embedding_lookup = {
        row['filename']: row.drop('filename').values
        for idx, row in embed_df.iterrows()
    }
    for m_index, m_row in data.iterrows():  # pylint: disable=unused-variable
        embedding = embedding_lookup[m_row['segment']]
        if 0 <= m_row['fold'] <= 3:
            x_train.append(embedding)
            if 0 <= m_row['label'] <= 4:
                y_train.append(1)
            else:
                y_train.append(0)
        else:
            x_test.append(embedding)
            if 0 <= m_row['label'] <= 4:
                y_test.append(1)
            else:
                y_test.append(0)
        print(f"added segment: {m_row['segment']} to dataset")
    for i, item in enumerate(x_train):
        if not isinstance(item, (np.ndarray, list)):
            print(f"Item {i} is weird! Type: {type(item)}")
        else:
            continue

    x_train = np.array(x_train).astype(np.float32)
    y_train = np.array(y_train)
    x_test = np.array(x_test).astype(np.float32)
    y_test = np.array(y_test)

    return x_train, y_train, x_test, y_test


def make_svm(meta, embeddings_df):
    """Obtain embeddings, train test split, and create an SVM

    Args:
        meta (str): the metadata file containing fold and label id as an int

        embeddings_df (str): the path to your embeddings folds/files.

    Returns:
        svm (model): Support vector machine model.
    """
    x_train, y_train, x_test, y_test = make_x_and_y(meta, embeddings_df)
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


def main(meta, embeds, source, embed_df, model_file):
    """Main script to run

    Args:
        meta (str): The metadata file containing fold and label id as an int.

        embeds (str): The path to your embeddings folds/files.

        source (str): What format your embeddings are in (currently
                      either perch or birdnet).

        embed_df (str): If you have a premade dataframe with the keys
                             as the filename and the values as the embedding
                             list, use this argument and leave embeds and
                             source empty.

        model_file (str): Path to desired model output file, must be a .pkl.
    """
    metadata = pd.read_csv(meta, index_col=0)
    if embed_df is not True:
        if embeds is not True:
            print("Found no path to embeddings folder/file. Please add "
                  "-embeds arg when running script.")
            sys.exit()
        if source == 'birdnet':
            embeddings_df = obtain_birdnet_embeddings(embeds)
        elif source == 'perch':
            embeddings_df = obtain_perch_embeddings(embeds)
        else:
            print(f"Cannot create embeddings without knowing the source,"
                  " ensure you selected -source perch or birdnet")
            sys.exit()
    else:
        embeddings_df = pd.read_csv(embed_df)

    if model_file is not True:
        svm = make_svm(metadata, embeddings_df)
        save_out_model(svm, model_file)
    else:
        make_svm(metadata, embeddings_df)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(
        description='Input Directory Path'
    )
    PARSER.add_argument('-meta', type=str,
                        help='Path to fold metadata')
    PARSER.add_argument('-embeds', type=str, default=False,
                        help='Path to directory with embeddings files.')
    PARSER.add_argument('-source', type=str, default=False,
                        help='Source of embeddings (birdnet or perch).')
    PARSER.add_argument('-embed_df', type=str, default=False,
                        help='Path to your premade embeddings dataframe.')
    PARSER.add_argument('-model_file', type=str,
                        help='File name and location of saved model.pkl.')
    ARGS = PARSER.parse_args()
    main(ARGS.meta, ARGS.embeds, ARGS.source, ARGS.embed_df, ARGS.model_file)
