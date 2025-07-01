"""Assess Birdnet performance on Buowset.

This allows for assessing how well Birdnet performs on
buowset for burrowing owl/no burrowing owl, and for
the individual call types within our labeled data.

Usage:
    python3 buowset_assess_birdnet.py /path/to/birdnet/output/
    /path/to/buowset/metadata.csv
"""
import argparse
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
from sklearn.metrics import recall_score, f1_score 


def organize_birdnet_output(birdnet_results):
    """
    """
    birdnet_df = pd.read_pickle(birdnet_results)
    return birdnet_df

def merge_metadata(metadata, birdnet_df):
    """
    """
    meta = pd.read_csv(metadata, index_col=0)
    df_merged = meta.merge(birdnet_df, on='segment')
    df_merged = df_merged.drop(columns=['segment_duration_s', 'fold'])

    return df_merged

def assess_birdnet(merged_data):
    """
    """
    y_true = merged_data['label'].map({0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 0}).values
    y_pred = merged_data['bn_label'].values

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


def main(birdnet_results, metadata):
    """Assess birdnet.
    """
    print("Starting")
    print("Aggregating BirdNET results.")
    birdnet_df = organize_birdnet_output(birdnet_results)
    print(f"Aggregated {len(birdnet_df)} BirdNET results.")
    print(f"Matching ground truth labels to BirdNET results.")
    merged_data = merge_metadata(metadata, birdnet_df)
    print(merged_data)
    print("Comparing BirdNET labels to ground truth.")
    assess_birdnet(merged_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Input Directory Path'
        )
    parser.add_argument('birdnet_results',
                        type=str,
                        help='Path to Birdnet results for padded buowset.')
    parser.add_argument('metadata',
                        type=str,
                        help='Path to buowset metadata file.')
    args = parser.parse_args()
    main(args.birdnet_results, args.metadata)
