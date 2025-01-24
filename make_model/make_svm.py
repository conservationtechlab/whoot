"""Create Support Vector Machine Model.

This script takes the labeled embeddings file, randomly divides into
a train and test set, trains a linear 2-class SVM, outputs the metrics,
and saves the model to a file to be used later.

Example:

    $ python make_svm.py /path/to/labeled_embeddings.csv \\
      /path/to/desired/model/output.sav

"""

import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, recall_score, precision_score
import numpy as np
import os
import pickle


def main(labeled_embeddings, saved_model):
    """Create and save SVM.

    Create the SVM, output the metrics, and save the model.

    Args:
        labeled_embeddings (str): The path to folder with labeled embeddings.
        saved_model (str): The path to the model.

    """
    all_X = []
    all_Y = []
    for embeddings_file in os.listdir(labeled_embeddings):
        embeddings_path = os.path.join(labeled_embeddings, embeddings_file)
        df = pd.read_csv(embeddings_path)

        X = df.drop(['Chunk Start', 'Chunk End', 'Label'])
        y = df['Label']
        all_X.append(X)
        all_Y.append(y)
    X_combined = pd.concat(all_X, ignore_index=True)
    Y_combined = pd.concat(all_Y, ignore_index=True)

    X_train, X_test, y_train, y_test = train_test_split(X_combined, Y_combined, test_size=0.2, random_state=42)


    svm = SVC(class_weight='balanced', probability=True)
    svm.fit(X_train, y_train)

    y_pred_default = svm.predict(X_test)

    pickle.dump(svm, open(saved_model, 'wb'))

    print("Classification report with default threshold:")
    print(classification_report(y_test, y_pred_default))

    print("Classification report with custom threshold of -0.33:")
    print(classification_report(y_test, y_pred_custom_threshold))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Input CSV and model output'
        )
    parser.add_argument('labeled_embeddings',
                        type=str,
                        help='Directory path to labels with embeddings.')
    parser.add_argument('saved_model',
                        type=str,
                        help='Path to the saved model output.')
    args = parser.parse_args()
    main(args.labeled_embeddings, args,saved_model)
