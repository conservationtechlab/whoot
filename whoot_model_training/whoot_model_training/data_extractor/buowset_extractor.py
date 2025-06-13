"""
Inspired by https://github.com/UCSD-E4E/pyha-analyzer-2.0/tree/main/pyha_analyzer/extractors
Standardizes the format of the buowset dataset
"""

import argparse
import os

import numpy as np
from datasets import (
    load_dataset,
    Audio,
    DatasetDict,
    load_from_disk,
    ClassLabel,
    Sequence
)
from ..dataset import AudioDataset
from pyha_analyzer.extractors.birdset import one_hot_encode_ds_wrapper


def one_hot_encode(row, classes):
    one_hot = np.zeros(len(classes))
    one_hot[row["labels"]] = 1
    row["labels"] = np.array(one_hot, dtype=float)
    return row

"""_summary_
"""
def buowset_extractor(
    metadata_csv,
    parent_path,
    output_path,  # TODO what does output do?
    validation_fold=4,
    test_fold=3,
    sr=32_000,
    filepath="segment",
):    

    # if os.path.exists(output_path):
    #     ds = load_from_disk(output_path)
    #     return AudioDataset(ds)

    # Hugging face by default defines a train split
    ds = load_dataset("csv", data_files=metadata_csv)["train"]
    ds = ds.rename_column("label", "labels") #Convention here is labels

    # Convert to a uniform one_hot encoding for classes
    ds = ds.class_encode_column("labels")
    class_list = ds.features["labels"].names
    mutlilabel_class_label =  Sequence(ClassLabel(names=class_list))
    ds = ds.map(
        lambda row: one_hot_encode(row, class_list)
    ).cast_column("labels", mutlilabel_class_label)
    
    # Get audio into uniform format
    
    ds = ds.add_column(
        "audio", 
        [os.path.join(parent_path, file) for file in ds[filepath]]
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
