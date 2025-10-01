"""Standardizes the format of the buowset dataset.

Inspired by https://github.com/UCSD-E4E/pyha-analyzer-2.0/
    tree/main/pyha_analyzer/extractors

The idea being extractors is that they take raw data, and
format it into a uniform dataset format, AudioDataset

This way, it should be easier to define what a
common audio dataset format is between
parts of the codebase for training

Supports both multilabel and binary labels
"""

import os
from dataclasses import dataclass

import numpy as np
from datasets import (
    load_dataset,
    Audio,
    DatasetDict,
    ClassLabel,
    Sequence,
)
from ..dataset import AudioDataset


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


@dataclass
class BuowsetParams:
    """Parameters that describe the Buowset.

    Args:
        validation_fold (int): label for valid split
        test_fold (int): label for valid split
        sample_rate (int): sample rate of the data
        filepath (int): name of column in csv for filepaths
    """

    validation_fold = 4
    test_fold = 3
    sr = 32_000
    filepath = "segment"


def buowset_extractor(
    metadata_csv,
    parent_path,
    output_path,
    params: BuowsetParams = BuowsetParams()
):
    """Extracts raw data in the buowset format into an AudioDataset.

    Args:
        Metdata_csv (str): Path to csv containing buowset metadata
        parent_path (str): Path to the parent folder for all audio data.
            Note its assumed the audio filepath
            in the csv is relative to parent_path
        output_path (str): Path to where HF cache for this dataset should live
        validation_fold (int): which fold is considered the validation set
            Default 4
        test_fold (int): Which fold is considered the test set Default 3
        sr (int): Sample Rate of the audio files Default: 32_000
        filepath (str): Name of the column in the dataset containing
        the filepaths Default: segment

    Returns:
        (AudioDataset): See dataset.py, AudioDatasets are consider
        the universal dataset for the training pipeline.
    """
    # Hugging face by default defines a train split
    ds = load_dataset("csv", data_files=metadata_csv)["train"]
    ds = ds.rename_column("label", "labels")  # Convention here is labels

    # Convert to a uniform one_hot encoding for classes
    ds = ds.class_encode_column("labels")
    class_list = ds.features["labels"].names
    multilabel_class_label = Sequence(ClassLabel(names=class_list))
    ds = ds.map(lambda row: one_hot_encode(row, class_list)).cast_column(
        "labels", multilabel_class_label
    )

    # Get audio into uniform format
    ds = ds.add_column(
        "audio", [
            os.path.join(parent_path, file) for file in ds[params.filepath]
        ]
    )

    ds = ds.add_column("filepath", ds["audio"])
    ds = ds.cast_column("audio", Audio(sampling_rate=params.sr))

    # Create splits of the data
    test_ds = ds.filter(lambda x: x["fold"] == params.test_fold)
    valid_ds = ds.filter(lambda x: x["fold"] == params.validation_fold)
    train_ds = ds.filter(
        lambda x: (x[
            "fold"
        ] != params.test_fold) & (x["fold"] != params.validation_fold)
    )
    ds = AudioDataset(
        DatasetDict({"train": train_ds, "valid": valid_ds, "test": test_ds})
    )

    ds.save_to_disk(output_path)

    return ds


def binarize_data(row, target_col=0):
    """Convert a multilabel label into a binary one.

    Args:
        row (dict): an example of data
        target_col (int): which index is the label for no_buow

    Returns:
        row (dict): now with a binary label instead
    """
    row["labels"] = [row["labels"][target_col], 1 - row["labels"][target_col]]
    return row


def buowset_binary_extractor(
        metadata_csv,
        parent_path,
        output_path,
        target_col=0):
    """Extracts raw data in the buowset format into an AudioDataset.

    BUT only allows for two classes: no_buow, yes_buow

    Args:
        Metdata_csv (str): Path to csv containing buowset metadata
        parent_path (str): Path to the parent folder for all audio data.
            Note its assumed the audio filepath
            in the csv is relative to parent_path
        output_path (str): Path to where HF cache for this dataset should live
        validation_fold (int): which fold is considered the validation set
            Default 4
        test_fold (int): Which fold is considered the test set Default 3
        sr (int): Sample Rate of the audio files Default: 32_000
        target_col (int): label for no_buow

    Returns:
        (AudioDataset): See dataset.py, AudioDatasets are consider
        the universal dataset for the training pipeline.
    """
    # Use the original extractor to create a multilabeled dataset
    ads = buowset_extractor(
        metadata_csv,
        parent_path,
        output_path,
    )

    # Now we just need to convert labels from multilabel to
    # 0 or 1
    binary_class_label = Sequence(ClassLabel(names=["no_buow", "buow"]))
    for split in ads:
        ads[split] = (
            ads[split]
            .map(lambda row: binarize_data(row, target_col=target_col))
            .cast_column("labels", binary_class_label)
        )

    return ads
