"""Standardizes the format of the buowset dataset

Inspired by https://github.com/UCSD-E4E/pyha-analyzer-2.0/tree/main/pyha_analyzer/extractors

The idea being extractors is that they take raw data, and 
format it into a uniform dataset format, AudioDataset

This way, it should be easier to define what a common audio dataset format is between 
parts of the codebase for training
"""

import os

import numpy as np
from datasets import load_dataset, Audio, DatasetDict, ClassLabel, Sequence, load_from_disk
from ..dataset import AudioDataset

# MAKE LOG OF DATASET USED
def one_hot_encode(row: dict, classes: list):
    """One hot Encodes a list of labels
    Args:
        row (dict): row of data in a dataset containing a labels column
        classes: a list of classes
    """
    one_hot = np.zeros(len(classes))
    one_hot[row["labels"]] = 1
    row["labels"] = np.array(one_hot, dtype=float)
    return row

def buowset_extractor(
    metadata_csv,
    parent_path,
    output_path,  # TODO what does output do?
    validation_fold=4,
    test_fold=3,
    sr=32_000,
    filepath="segment",
):
    """Extracts raw data in the buowset format into an AudioDataset

    Args:
        Metdata_csv (str): Path to csv containing buowset metadata
        parent_path (str): Path to the parent folder for all audio data. 
            Note its assumed the audio filepath in the csv is relative to parent_path
        output_path (str): Path to where HF cache for this dataset should live
        validation_fold (int): which fold is considered the validation set Default 4
        test_fold (int): Which fold is considered the test set Default 3
        sr (int): Sample Rate of the audio files Default: 32_000
        filepath (str): Name of the column in the dataset containing the filepaths Default: segment

    Returns:
        (AudioDataset): See dataset.py, AudioDatasets are consider the universal dataset for the training pipeline. 
    """
    # if os.path.exists(output_path):
    #     ds = load_from_disk(output_path)
    #     return AudioDataset(ds)

    # Hugging face by default defines a train split
    ds = load_dataset("csv", data_files=metadata_csv)["train"]
    ds = ds.rename_column("label", "labels")  # Convention here is labels
    

    # Convert to a uniform one_hot encoding for classes
    ds = ds.class_encode_column("labels")
    class_list = ds.features["labels"].names
    mutlilabel_class_label = Sequence(ClassLabel(names=class_list))
    ds = ds.map(lambda row: one_hot_encode(row, class_list)).cast_column(
        "labels", mutlilabel_class_label
    )

    # Get audio into uniform format
    ds = ds.add_column(
        "audio", [os.path.join(parent_path, file) for file in ds[filepath]]
    )

    ds = ds.cast_column("audio", Audio(sampling_rate=sr))

    # Create splits of the data
    test_ds = ds.filter(lambda x: x["fold"] == validation_fold)
    valid_ds = ds.filter(lambda x: x["fold"] == test_fold)
    train_ds = ds.filter(
        lambda x: x["fold"] != test_fold & x["fold"] != validation_fold
    )
    ds = AudioDataset(
        DatasetDict({"train": train_ds, "valid": valid_ds, "test": test_ds})
    )

    ds.save_to_disk(output_path)

    return ds



def binarize_data(row, target_col=0):
    row["labels"] = [row["labels"][target_col], 1-row["labels"][target_col]]
    return row

def buowset_binary_extractor(
        metadata_csv,
        parent_path,
        output_path,  # TODO what does output do?
        validation_fold=4,
        test_fold=3,
        sr=32_000,
        filepath="segment",
        target_col = 0
    ):


    ads = buowset_extractor(metadata_csv,
        parent_path,
        output_path,
        validation_fold=validation_fold,
        test_fold=test_fold,
        sr=sr,
        filepath=filepath
    )

    binary_class_label = Sequence(ClassLabel(names=["no_buow", "buow"]))
    print(binary_class_label.feature.num_classes)
    for split in ads:
        ads[split] = ads[split].map(lambda row: binarize_data(row, target_col=target_col)).cast_column(
            "labels", binary_class_label
        )
    
    print(ads.get_num_classes())

    return ads