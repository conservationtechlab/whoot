"""Make SVM with buowset data

This script can be run to create an SVM from buowset data.

Usage: python3 make_svm.py -meta /path/to/fold/metadata.csv
    -embeds /path/to/birdnet/embedding/directory/

"""
import pandas as pd
import argparse
import pickle
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import os
import numpy as np
import csv


def make_x_and_y(data, embeds):
    """Custom train/test split based on folds

    The buowset data is in 5 stratified group folds.
    This means we must create train test splits ourselves,
    because they are not random but based on creating 5
    groups with a relatively equal class distribution
    among them. This current version uses the 5th fold
    as the test set and separates all burrowing owl
    vocalizations from no_buow class.

    # TODO: Allow user to easily select which fold will be
            the test set, and which classes are x and y

    Args:
        data (pd.Dataframe): The metadata with the fold and class info.

        embeds (str): The path to the folder containing all birdnet embeddings.
    """
    x_train = [] # 4 of the folds
    y_train = [] # 4 of the folds
    x_test = [] # the small one 20%, 1 folds worth
    y_test = [] # the small one 20%, 1 folds worth
    for index, row in data.iterrows():
        filename = row['segment']
        filename = filename.replace(".wav", "")
        embed_name = filename + ".birdnet.embeddings.txt"
        embedpath = os.path.join(embeds, embed_name)
        dfb = pd.read_csv(embedpath,
                          delimiter="[,\t]",
                          engine='python',
                          header=None)
        dfb_stripped = dfb.drop(dfb.columns[:2], axis=1)
        flattened = dfb_stripped.values.flatten()
        if len(flattened) > 1024:
            print(f"filename {filename} has extra lines. Trunicating")
            flattened = flattened[:1024]
        if 0 <= row['fold'] <= 3:
            x_train.append(flattened)
            if 0 <= row['label'] <= 4:
                y_train.append(1)
            else:
                y_train.append(0)
        else:
            x_test.append(flattened)
            if 0 <= row['label'] <= 4:
                y_test.append(1)
            else:
                y_test.append(0)
        print(f"added segment: {filename} to dataset")

    x_train = np.vstack(x_train).astype(np.float16)
    y_train = np.array(y_train)
    x_test = np.vstack(x_test).astype(np.float16)
    y_test = np.array(y_test)

    return x_train, y_train, x_test, y_test

def make_svm(meta, embeds):
    """Make the svm and save it out

    This script takes the custom split train and test sets
    and trains an SVM. It prints the classification report
    and saves the model.

    Args:
        meta (str): The metadata with the fold and class info.

        embeds (str): The path to the folder containing all birdnet embeddings.
    """
    data = pd.read_csv(meta, index_col=0)
    x_train, y_train, x_test, y_test = make_x_and_y(data, embeds)
    print("beginning model training")
    svm = SVC(class_weight='balanced', probability=True)
    svm.fit(x_train, y_train)

    y_pred_default = svm.predict(x_test)
    saved_model = 'model.pkl'
    with open(saved_model, 'wb') as file:
        pickle.dump(svm, file)
    print("Classification report with default threshold:")
    print(classification_report(y_test, y_pred_default))

def main(meta, embeds):
    """Main function

    Args:
        meta (str): The metadata with the fold and class info.

        embeds (str): The path to the folder containing all birdnet embeddings.
    """
    make_svm(meta, embeds)

if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description='Input Directory Path'
    )
    parser.add_argument('-meta', type=str,
                        help='Path to fold metadata')
    parser.add_argument('-embeds', type=str,
                        help='Path to directory with embeddings files')
    args = parser.parse_args()
    main(args.meta, args.embeds)

