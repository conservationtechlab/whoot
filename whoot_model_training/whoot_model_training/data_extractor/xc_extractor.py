"""Ceates Dataset from the Xeno-Canto Data Downlaoder tool.

See data_downloader/xc.py
"""

import os
import shutil
from pathlib import Path
from dataclasses import dataclass
from collections import Counter
from pydub import AudioSegment


import numpy as np
from datasets import (
    Dataset,
    Audio,
    DatasetDict,
    ClassLabel,
    Sequence,
    load_from_disk,
)
from ..dataset import AudioDataset

import json
import librosa


def filter_by_count(ds, col="en", threshold=10):
    """Limit species list to species with some amount of species."""
    count_by_species = Counter(ds[col])
    return ds.filter(
        lambda row: count_by_species[row] > threshold,
        input_columns=[col]
    )


def filter_xc_data(row: dict):
    """In personal experience, raw XC data is very messy.

    Some files get coruptted
    This intention checks to see if loading files is
    possible for the frist place
    """
    file_path = row["filepath"]
    try:
        # Heuristic, if we can load 3 seconds, file is probably okay
        # Prevents some files from taking forever
        librosa.load(path=file_path, duration=3)
        return True
    except Exception as e:
        print(e, file_path)
        return False


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


def convert_audio_to_flac(row, error_path="bad_files", col="audio"):
    """Convert any audio to flac for better compression.

    Args:
        row: row from hugging face table
        error_path: folder to dump broken files
        col: column with audio path
    """
    file_path = row[col]
    flac_path = Path(file_path).parent / (Path(file_path).stem + ".flac")
    if os.path.exists(flac_path):
        row[col] = str(flac_path)
        if os.path.exists(file_path):
            os.remove(file_path)  # Remove origional file, we don't need it
        return row
    try:
        wav_audio = AudioSegment.from_file(file_path)
        wav_audio.export(flac_path, format="flac")
    except Exception as e:
        if os.path.exists(file_path):
            os.makedirs(error_path, exist_ok=True)
            shutil.move(file_path, error_path)
        # If quit halfway through processing,
        # make sure we get rid of the bad file
        # if os.path.exists(flac_path):
        #     os.remove(flac_path)
        print(
            "ERROR",
            "move to",
            os.path.join(error_path, Path(file_path).name),
            "ERR MSG:",
            e
        )
        row[col] = str(os.path.join(error_path, Path(file_path).name))
        return row
    row[col] = str(flac_path)
    return row


@dataclass
class XCParams():
    """Parameters that describe ESC-50.

    validation_fold (int): label for valid split
    test_fold (int): label for valid split
    sample_rate (int): sample rate of the data
    filepath (string): name of column in csv for filepaths
    """
    validation_fold = 4
    test_fold = 5
    sample_rate = 44_100


def xc_extractor(
        XC_dataset_json_path,
        parent_path,
        cache_path="data/san_diego_xc_aux/cache",
        params: XCParams = XCParams(),
        bad_file_path="data/xc_bad_file"
):
    """Extracts data collected from the XC downloader.

    XC_dataset_json_path: json outputted from XC downloader
    parent_path: path to highest level audio file
    cache_path: path to cache hugging
    """
    if os.path.exists(cache_path):
        return load_from_disk(cache_path)

    with open(XC_dataset_json_path, mode="r") as f:
        xc_recordings_paged = json.load(f)

    xc_recordings = []
    for page in xc_recordings_paged:
        xc_recordings.extend(page["recordings"])

    dataset = Dataset.from_list(xc_recordings)

    dataset = dataset.add_column("labels", dataset["en"])
    dataset = dataset.class_encode_column("labels")
    class_list = dataset.features["labels"].names
    multilabel_class_label = Sequence(ClassLabel(names=class_list))
    dataset = dataset.map(
        lambda row: one_hot_encode(row, class_list)
    ).cast_column(
        "labels",
        multilabel_class_label
    )

    dataset = dataset.add_column(
        "audio", [
            os.path.join(
                parent_path,
                file.replace("/", "_")
            ) for file in dataset["file-name"]
        ]
    )

    # Fix file paths
    dataset = dataset.map(
        convert_audio_to_flac,
        fn_kwargs={"error_path": bad_file_path},
        num_proc=16
    )
    dataset = dataset.filter(
        lambda x: bad_file_path not in x["audio"], num_proc=16
    )
    dataset = dataset.add_column("filepath", dataset["audio"])
    dataset = dataset.cast_column(
        "audio",
        Audio(sampling_rate=params.sample_rate)
    )

    # TODO FIGURE OUT HOW TO DO SPLITS!
    # # Create splits of the data
    # test_ds = dataset.filter(lambda x: x["fold"] == params.test_fold)
    # valid_ds = dataset.filter(lambda x: x["fold"] == params.validation_fold)
    # train_ds = dataset.filter(
    #     lambda x: (
    #         x["fold"] != params.test_fold
    #         and x["fold"] != params.validation_fold
    #     )
    # )

    dataset = dataset.cast_column(
        "en", ClassLabel(names=list(set(dataset["en"])))
    )

    dataset = filter_by_count(dataset)

    train_test = dataset.train_test_split(0.2, stratify_by_column="en")
    test_val = train_test["test"].train_test_split(
        0.2,
        stratify_by_column="en"
    )

    dataset = AudioDataset(
        DatasetDict({
            "train": train_test["train"],
            "valid": test_val["train"],
            "test": test_val["test"]})
    )

    # os.makedirs(cache_path, exist_ok=True)
    # dataset.save_to_disk(cache_path)

    return dataset
