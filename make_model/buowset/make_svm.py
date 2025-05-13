"""Make SVM with buowset data

This script can be run to create an SVM from buowset data.

Usage: python3 make_svm.py -meta /path/to/fold/metadata.csv
    -embeds /path/to/embedding/directory/or/file -source birdnet (or perch)
    -model_file /path/of/model.pkl

"""
import argparse
import os
import glob
import ntpath
import pickle
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import numpy as np


def obtain_perch_embeddings(embeds):
    """Create dict dataframe with filename and embedding list
    """
    # placeholder for actual function
    embeddings_df = embeds

    return embeddings_df


def obtain_birdnet_embeddings(embeds):
    """Create a dict dataframe with filename and embedding list

    Args:
        embeds (str): Path to directory where embeddings files are.

    Returns:
        embed_df (pd.Dateframe): A dictonary with the filename as the
                                 key and the list of floats (embeddings)
                                 as the value
    """
    embed_dict = {}
    text_files = glob.glob(os.path.join(embeds, "*.txt"))
    for embed in text_files:
        filename = ntpath.basename(embed)
        filename = filename.replace(".birdnet.embeddings.txt", ".wav")
        dfb = pd.read_csv(embed,
                          delimiter="[,\t]",
                          engine='python',
                          header=None)
        dfb_stripped = dfb.drop(dfb.columns[:2], axis=1)
        flattened = dfb_stripped.values.flatten()
        if len(flattened) > 1024:
            print(f"filename {filename} has extra lines. Trunicating")
            flattened = flattened[:1024]
        embed_dict[filename] = flattened

    embed_df = pd.DataFrame({
        'filename': list(embed_dict.keys()),
        'embeddings': list(embed_dict.values())
    })
    # for debug
    embed_df.to_csv("birdnet_df_embed_with_filename.csv",
                    encoding='utf-8', index=False)
    return embed_df


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
    for m_index, m_row in data.iterrows():  # pylint: disable=unused-variable
        embedding = embed_df.loc[embed_df['filename'] == m_row['segment'],
                                 'embeddings'].values[0]
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


def make_svm(meta, embeds, source, model_file):
    """Obtain embeddings, train test split, and create an SVM

    Args:
        meta (str): the metadata file containing fold and label id as an int

        embeds (str): the path to your embeddings folds/files

        source (str): what format your embeddings are in (currently
                      either perch or birdnet)

        model_file (str): Path to desired model output file, must be a .pkl
    """
    data = pd.read_csv(meta, index_col=0)
    embeddings_df = None
    if source == 'birdnet':
        embeddings_df = obtain_birdnet_embeddings(embeds)
    elif source == 'perch':
        embeddings_df = obtain_perch_embeddings(embeds)
    else:
        print(f"Can't obtain embeddings, ensure you selected perch or birdnet")
    x_train, y_train, x_test, y_test = make_x_and_y(data, embeddings_df)
    print("beginning model training")
    svm = SVC(class_weight='balanced', probability=True)
    svm.fit(x_train, y_train)

    y_pred_default = svm.predict(x_test)
    with open(model_file, 'wb') as file:
        pickle.dump(svm, file)
    print("Classification report with default threshold:")
    print(classification_report(y_test, y_pred_default))


def main(meta, embeds, source, model_file):
    """Main script to run

    Args:
        meta (str): The metadata file containing fold and label id as an int.

        embeds (str): The path to your embeddings folds/files.

        source (str): What format your embeddings are in (currently
                      either perch or birdnet).

        model_file (str): Path to desired model output file, must be a .pkl.
    """
    make_svm(meta, embeds, source, model_file)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(
        description='Input Directory Path'
    )
    PARSER.add_argument('-meta', type=str,
                        help='Path to fold metadata')
    PARSER.add_argument('-embeds', type=str,
                        help='Path to directory with embeddings files')
    PARSER.add_argument('-source', type=str,
                        help='Source of embeddings (birdnet or perch)')
    PARSER.add_argument('-model_file', type=str,
                        help='File name and location of saved model.pkl')
    ARGS = PARSER.parse_args()
    main(ARGS.meta, ARGS.embeds, ARGS.source, ARGS.model_file)
