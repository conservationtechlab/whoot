"""Assess Birdnet Metrics.

This script compares human and birdnet labels and outputs the
confusion matrix, accuracy, precision, recall, and F1 score
assuming the human labels are 100% accurate.

Example:

    $ python assess_performance.py /path/to/human_labeled.csv \\
      /path/to/birdnet_labeled.csv

"""

import argparse
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
from sklearn.metrics import recall_score, f1_score


def main(human_labeled, birdnet_labeled):
    """Evaluate Birdnet Metrics.

    Main script that prints metrics comparing human/birdnet labeled
    acoustic data assuming the human labels are ground truth.

    Args:
        human_labeled (str): The path to the human labeled csv.
        birdnet_labeled (str): The path to the adjusted birdnet output.

    """
    scored_data = pd.read_csv(human_labeled)
    ml_output = pd.read_csv(birdnet_labeled)

    y_true = scored_data['Label'].map({'yes': 1, 'no': 0}).values
    y_pred = ml_output['Label'].map({'yes': 1, 'no': 0}).values

    confusion_m = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1_result = f1_score(y_true, y_pred)

    print("Confusion Matrix:")
    print(confusion_m)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_result:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Input Directory Path'
        )
    parser.add_argument('human_labeled',
                        type=str,
                        help='Path to human labeled adjusted output.')
    parser.add_argument('birdnet_labeled',
                        type=str,
                        help='Path to birdnet labeled adjusted output.')
    args = parser.parse_args()
    main(args.human_labeled, args.birdnet_labeled)
