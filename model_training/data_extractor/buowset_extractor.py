"""
Inspired by https://github.com/UCSD-E4E/pyha-analyzer-2.0/tree/main/pyha_analyzer/extractors
Standardizes the format of the buowset dataset
"""

import argparse
import os
from datasets import load_dataset, Audio, DatasetDict
from ..dataset import AudioDataset

"""_summary_
"""


def buowset_extractor(
    metadata_csv,
    parent_path,
    output,  # TODO what does output do?
    validation_fold=4,
    test_fold=3,
    sr=32_000,
    filepath="segment",
):
    ds = load_dataset(metadata_csv)
    ds["audio"] = parent_path + "/" + ds[filepath]  # TODO Better file path handling pls
    ds = ds.cast_column("audio", Audio(sampling_rate=sr))

    test_ds = ds.filter(lambda x: x["fold"] == validation_fold)
    valid_ds = ds.filter(lambda x: x["fold"] == test_fold)
    train_ds = ds.filter(
        lambda x: x["fold"] != test_fold & x["fold"] != validation_fold
    )

    return AudioDataset(
        DatasetDict({"train": train_ds, "valid": valid_ds, "test_ds": test_ds})
    )
