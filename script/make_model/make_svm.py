"""Create Support Vector Machine Model.

This script takes the labeled embeddings file, randomly divides into
a train and test set, trains a linear 2-class SVM, outputs the metrics,
and saves the model to a file to be used later.

Example:

    $ python make_svm.py /path/to/labeled_embeddings.csv \
      /path/to/desired/model/output.sav

"""

import argparse
import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report


def main(labeled_embeddings, saved_model):
    """Create and save SVM.

    Create the SVM, output the metrics, and save the model.

    Args:
        labeled_embeddings (str): The path to folder with labeled embeddings.
        saved_model (str): The path to the model.

    """
    all_x = []
    all_y = []
    for embeddings_file in os.listdir(labeled_embeddings):
        embeddings_path = os.path.join(labeled_embeddings, embeddings_file)
        le_df = pd.read_csv(embeddings_path)
        embed = le_df.drop(['Chunk Start', 'Chunk End', 'Label'], axis=1)
        label = le_df['Label']
        all_x.append(embed)
        all_y.append(label)
    combined_x = pd.concat(all_x, ignore_index=True)
    combined_y = pd.concat(all_y, ignore_index=True)

    print(f"Detection types in entire set: \n{combined_y.value_counts()}")

    train_x, test_x, train_y, test_y = train_test_split(combined_x,
                                                        combined_y,
                                                        test_size=0.2,
                                                        random_state=42)

    svm = SVC(class_weight='balanced', probability=True)
    svm.fit(train_x, train_y)

    y_pred_default = svm.predict(test_x)

    with open(saved_model, 'wb') as file:
        pickle.dump(svm, file)

    print("Classification report with default threshold:")
    print(classification_report(test_y, y_pred_default))


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
    main(args.labeled_embeddings, args.saved_model)
