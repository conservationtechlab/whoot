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
from comet_ml import Experiment
import random


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

def map_binary_labels(merged_data):
    """Obtain the two dataframes for the predicted and true labels.

    Args:
    """
    y_true = merged_data['label'].map({0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 0}).values
    y_pred = merged_data['bn_label'].values

    return y_true, y_pred


def map_class_labels(merged_data, assess_class):
    """
    """
    class_only = merged_data[merged_data['label'] == assess_class]
    num_rows = len(class_only) - 1
    no_buow_only = merged_data[merged_data['label'] == 5]
    num_no_buow_rows = len(no_buow_only) - 1
    available_numbers = list(range(0, num_rows))
    available_indexes = list(range(0, num_no_buow_rows))
    index_no_buow = []
    index = 0
    while available_numbers:
        selected_number = random.choice(available_indexes)
        index_no_buow.append(selected_number)
        available_indexes.remove(selected_number)
        available_numbers.remove(index)
        index += 1
    no_buow_subset = no_buow_only.iloc[index_no_buow]
    merged = pd.concat([no_buow_subset, class_only], ignore_index=True)
    y_true = merged['label'].map({assess_class: 1, 5: 0}).values
    y_pred = merged['bn_label'].values

    return y_true, y_pred


def assess_birdnet(y_true, y_pred, experiment=None):
    """Assess Birdnet against ground truth labels.

    Args:
    """
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

    if experiment:
        experiment.log_metric("accuracy", accuracy)
        experiment.log_metric("precision", precision)
        experiment.log_metric("recall", recall)
        experiment.log_metric("f1_score", f1_result)
        experiment.log_confusion_matrix(matrix=confusion_m.tolist(),
                                        labels=["No Detection", "Detection"])


def create_comet_exp():
    """
    """
    project = input("Enter the comet project name you'd like this experiment to have/be associated with: ")
    work_space = input("Enter the comet workspace (username or organization) this experiment will go in: ")
    experiment_name = input("Enter the name of this experiment: ")
    experiment = Experiment(
        project_name=project,
        workspace=work_space
    )
    experiment.set_name(experiment_name)
    experiment.add_tags(["burrowl", "birdnet", "binary-classification"])

    return experiment

def main(birdnet_results, metadata, not_binary, assess_class):
    """Assess birdnet.
    """
    print("Starting")
    experiment = create_comet_exp()
    print("Aggregating BirdNET results.")
    birdnet_df = organize_birdnet_output(birdnet_results)
    print(f"Aggregated {len(birdnet_df)} BirdNET results.")
    print(f"Matching ground truth labels to BirdNET results.")
    merged_data = merge_metadata(metadata, birdnet_df)
    print("Comparing BirdNET labels to ground truth.")
    if not_binary == True:
        print("Doing binary buow/no_buow assessment")
        y_true, y_pred = map_binary_labels(merged_data)
    else:
        print(f"Assessing performance of Birdnet on vocalization: {assess_class}")
        y_true, y_pred = map_class_labels(merged_data, assess_class)
    assess_birdnet(y_true, y_pred, experiment=experiment)


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
    parser.add_argument('-not_binary', action='store_false',
                        help='Default true binary assessment, call for individual class assessment.')
    parser.add_argument('-assess_class', default=None, type=int,
                        help='Which class would you like to assess individually?')
    args = parser.parse_args()
    main(args.birdnet_results, args.metadata, args.not_binary, args.assess_class)
