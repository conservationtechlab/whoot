"""Processes raw audio folders.

Extractor for general, typically unlabeled soundscape recordings

Fits as much as possible to the AudioDataset standard but 
NOT INTENDED FOR TRAINING

Rather just a placeholder to help inferance work
"""

import os
from dataclasses import dataclass
import glob

import numpy as np
from datasets import (
    load_dataset,
    Audio,
    DatasetDict,
    ClassLabel,
    Sequence,
)
from ..dataset import AudioDataset

# class AudioChunk(Audio):
#     def __init__(self, offsets, duration, *args, *kwargs):
#         self.offsets = offsets
#         self.duration = duration
#         return super().__init__(*args, *kwargs)


def one_hot_encode(row: dict, classes: list):
    """One hot Encodes a list of labels.

    Args:
        row (dict): row of data in a dataset containing a labels column
        classes: a list of classes
    """
    print(row["labels"])
    one_hot = np.zeros(len(classes))
    one_hot[row["labels"]] = 1
    row["labels"] = np.array(one_hot, dtype=int)
    return row

def raw_audio_extractor(
    audio_parent_folder,
    sr=32_000,
    class_list = ["cluck", "coocoo", "twitter", "alarm", "chick begging", "no_buow"]
):
    """Extracts raw data in the buowset format into an AudioDataset.

    Args:
        audio_parent_folder (str): Path to the parent folder for all audio data.
            Note its assumed the audio filepath
            in the csv is relative to parent_path
        sr (int): Sample Rate of the audio files Default: 32_000

    Returns:
        (AudioDataset): See dataset.py, AudioDatasets are consider
        the universal dataset for the training pipeline.
    """

    dataset = load_dataset("audiofolder", data_dir=audio_parent_folder)
    dataset["train"] = dataset["train"].add_column("labels", np.zeros(dataset["train"].shape[0]).astype(int))

    # # Convert to a uniform one_hot encoding for classes
    dataset = dataset.class_encode_column("labels")
    multilabel_class_label = Sequence(ClassLabel(names=class_list))
    dataset = dataset.map(lambda row: one_hot_encode(row, class_list)).cast_column(
        "labels", multilabel_class_label
    )

    ds = AudioDataset(
        DatasetDict({"train": dataset["train"], "valid": dataset["train"], "test": dataset["train"]})
    )
    return ds

