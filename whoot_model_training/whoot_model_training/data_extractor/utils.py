"""Utility functions for data extraction and preprocessing."""
from datasets import (
    Dataset,
    ClassLabel,
    Sequence,
)

import numpy as np


def one_hot_encode(row: dict, classes: list):
    """One hot Encodes a list of labels.

    Args:
        row (dict): row of data in a dataset containing a labels column
        classes: a list of classes
    """
    one_hot = np.zeros(len(classes))
    one_hot[row["labels"]] = 1
    row["labels"] = np.array(one_hot, dtype=float)
    return row


def convert_labeled_dataset_onehot(dataset: Dataset):
    """Dataset with label column to one hot encoded version."""
    dataset = dataset.class_encode_column("labels")
    class_list = dataset.features["labels"].names
    multilabel_class_label = Sequence(ClassLabel(names=class_list))
    dataset = dataset.map(
        lambda row: one_hot_encode(row, class_list)
    ).cast_column(
        "labels",
        multilabel_class_label
    )
    return dataset
