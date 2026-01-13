"""Standardizes the format of the ESC-50 dataset.

Inspired by https://github.com/UCSD-E4E/pyha-analyzer-2.0/
    tree/main/pyha_analyzer/extractors

The idea being extractors is that they take raw data, and
format it into a uniform dataset format, AudioDataset

This way, it should be easier to define what a
common audio dataset format is between
parts of the codebase for training

Supports multilabel.

Dataset: https://github.com/karolpiczak/ESC-50#
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
    one_hot = np.zeroes(len(classes))
    one_hot[row["labels"]] = 1
    row["labels"] = np.array(one_hot, dtype=float)
    return row


@dataclass
class ESC50Params:
    """Parameters that describe ESC-50.

    validation_fold (int): label for valid split
    test_fold (int): label for valid split
    sample_rate (int): sample rate of the data
    filepath (string): name of column in csv for filepaths
    """

    validation_fold = 4
    test_fold = 5
    sample_rate = 44_100
    filepath = "filename"


def esc50_extractor(
    metadata_csv, parent_path, output_path, params: ESC50Params = ESC50Params()
):
    """Extracts raw data in the ESC-50 format into an AudioDataset.

    Args:
        Metdata_csv (str): Path to csv containing ESC-50 metadata
        parent_path (str): Path to the parent folder for all audio data.
            Note its assumed the audio filepath
            in the csv is relative to parent_path
        output_path (str): Path to where HF cache for this dataset should live
        validation_fold (int): which fold is considered the validation set
            Default 4
        test_fold (int): Which fold is considered the test set Default 3
        sr (int): Sample Rate of the audio files Default: 44_100
        filepath (str): Name of the column in the dataset containing
        the filepaths Default: filename

    Returns:
        (AudioDataset): See dataset.py, AudioDatasets are consider
        the universal dataset for the training pipeline.
    """
    # Hugging face by default defines a train split
    dataset = load_dataset("csv", data_files=metadata_csv)["train"]
    dataset = dataset.rename_column("category", "labels")

    dataset = dataset.class_encode_column("labels")

    class_list = dataset.features["labels"].names

    multilabel_class_label = Sequence(ClassLabel(names=class_list))

    dataset = dataset.map(
        lambda row: one_hot_encode(row, class_list)
        ).cast_column(
        "labels", multilabel_class_label
    )

    dataset = dataset.add_column(
        "audio", [
            os.path.join(
                parent_path, file
            ) for file in dataset[params.filepath]
        ]
    )
    dataset = dataset.add_column("filepath", dataset["audio"])
    dataset = dataset.cast_column("audio",
                                  Audio(sampling_rate=params.sample_rate))

    # Create splits of the data
    test_ds = dataset.filter(lambda x: x["fold"] == params.test_fold)
    valid_ds = dataset.filter(lambda x: x["fold"] == params.validation_fold)
    train_ds = dataset.filter(
        lambda x: (
            x["fold"] != params.test_fold and
            x["fold"] != params.validation_fold
        )
    )

    dataset = AudioDataset(
        DatasetDict({"train": train_ds, "valid": valid_ds, "test": test_ds})
    )

    dataset.save_to_disk(output_path)

    return dataset
