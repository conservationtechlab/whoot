"""Processes raw audio folders.

Extractor for general, typically unlabeled soundscape recordings

Fits as much as possible to the AudioDataset standard but
NOT INTENDED FOR TRAINING

Rather just a placeholder to help inferance work
"""

from typing import Any, ClassVar, Union
import os
from math import floor

import numpy as np
from datasets import (
    Audio,
    concatenate_datasets,
    DatasetDict,
    ClassLabel,
    Sequence,
    Dataset,
    table,
)
from datasets.features.features import _FEATURE_TYPES, FeatureType

import librosa
from tqdm import tqdm
import pyarrow as pa
from ..dataset import AudioDataset


class SubAudio(Audio):
    """Extends Audio to take a chunks of data.

    Uses code from the Hugging Face Audio Class
    https://github.com/huggingface/datasets/blob/5dc1a179783dff868b0547c8486268cfaea1ea1f/src/datasets/features/audio.py#L24

    The Audio Column of a HuggingFace dataset
    handles loading in data from a given file

    What is nice is it streams data: it doesn't get loaded into
    memory until it is needed via the path

    However, if we wanted to load in a chunk of data (some segment)
    We would need to load it as an array instead of a path
    And it gets loaded into memory. Huge issue with large audio datasets.

    By default HF doesn't support chunking,
    so this class should handle chunking
    During streaming rather than during dataset creation

    You can use it the same way you might with the Audio class. In fact,
    with normal processing
    it handles the same way!

    To use the chunking feature, create a Audio row with the following
    parameters
    - path: as is with Audio
    - sampling_rate: as is with Audio
    - offset: NEW, offset in seconds of when to start taking audio data
    - duration: NEW, duration from offset in seconds for how much data to
    collect

    You need both offset and duration to load in the chunk,
    otherwise it will load the full file.
    """

    pa_type: ClassVar[Any] = pa.struct(
        {
            "bytes": pa.binary(),
            "path": pa.string(),
            "offset": pa.int64(),
            "duration": pa.int64(),
        }
    )

    def __call__(self):
        """Return the type of SubAudio."""
        return self.pa_type

    def encode_example(self, value) -> dict:
        """Encodes an audio into bytes."""
        if (
            isinstance(value, dict)
            and value.get("offset")
            and value.get("duration")
            and value.get("path") is not None
            and os.path.isfile(value["path"])
        ):
            y, sr = librosa.load(
                path=value["path"],
                offset=value["offset"],
                duration=value["duration"]
            )
            value["array"] = y
            value["sampling_rate"] = sr
            encoded = super().encode_example(value)
            encoded["offset"] = value["offset"]
            encoded["duration"] = value["duration"]
            encoded["path"] = encoded["path"]
            return encoded
        return super().encode_example(value)

    def decode_example(self, value, token_per_repo_id=None) -> dict:
        """Decodes an encoded value.

        This will only return a segment of the data
        """
        # This is how this function works in the normal
        # Audio type
        if (
            # pylint: disable=too-many-boolean-expressions
            isinstance(value, dict)
            and "offset" in value
            and "duration" in value
            and value.get("bytes") is None
            and value.get("path") is not None
            and os.path.isfile(value["path"])
        ):
            y, sr = librosa.load(
                path=value["path"],
                offset=value["offset"],
                duration=value["duration"]
            )
            return {
                "path": value["path"],
                "array": y,
                "sampling_rate": sr,
                "offset": value["offset"],
                "duration": value["duration"],
            }
        if (
            isinstance(value, dict)
            and value.get("offset")
            and value.get("duration")
            and value.get("bytes") is not None
        ):
            decoded = super().decode_example(
                value,
                token_per_repo_id=token_per_repo_id
            )
            decoded["offset"] = value["offset"]
            decoded["duration"] = value["duration"]
            return decoded

        return super().decode_example(
            value,
            token_per_repo_id=token_per_repo_id
        )

    def cast_storage(
        self, storage: Union[pa.StringArray, pa.StructArray]
    ) -> pa.StructArray:
        """Cast a column in a Hugging Face dataset."""
        # print("cast_storage real")
        if pa.types.is_struct(storage.type):
            if storage.type.get_field_index("bytes") >= 0:
                bytes_array = storage.field("bytes")
            else:
                bytes_array = pa.array([None] * len(storage), type=pa.binary())
            if storage.type.get_field_index("path") >= 0:
                path_array = storage.field("path")
            else:
                path_array = pa.array([None] * len(storage), type=pa.string())
            if storage.type.get_field_index("offset") >= 0:
                offset_array = storage.field("offset")
            else:
                offset_array = pa.array([None] * len(storage), type=pa.int64())
            if storage.type.get_field_index("duration") >= 0:
                duration_array = storage.field("duration")
            else:
                duration_array = pa.array([None] * len(storage),
                                          type=pa.int64())
            storage = pa.StructArray.from_arrays(
                [bytes_array, path_array, offset_array, duration_array],
                ["bytes", "path", "offset", "duration"],
                mask=storage.is_null(),
            )
        return table.array_cast(storage, self.pa_type)


_FEATURE_TYPES[SubAudio.__name__] = SubAudio
FeatureType = Union[FeatureType, SubAudio]


def get_empty_dict():
    """Get a sample dict for saving audio."""
    return {
        "audio": [],
        "file_path": [],
        "labels": [],
    }


def get_array_chunks_from_memory(
    parent_folder,
    chunk_length_sec=5,
    no_class_idx=5,
):
    """Get audio chunks."""
    new_rows = get_empty_dict()
    _datasets = []
    for root, _, files in tqdm(os.walk(parent_folder), desc="All Folders"):
        for filename in tqdm(files, leave=False, desc="file in dir"):

            if not filename.lower().endswith(
                (".wav", ".mp3", ".flac", ".ogg", ".m4a")
            ):
                continue
            file_path = os.path.join(root, filename)
            try:
                clip_length = librosa.get_duration(path=file_path)
                # sr = librosa.get_samplerate(path=file_path)
            except IOError as e:
                print(e, file_path, "failed stat read", "continuing")
                continue
            for i in tqdm(
                range(0, int(floor(clip_length)), chunk_length_sec),
                leave=False,
                desc=f"{filename}",
            ):
                new_rows["audio"].append(
                    {
                        "path": file_path,
                        # "sampling_rate": sr,
                        "offset": i,
                        "duration": chunk_length_sec,
                    }
                )
                new_rows["file_path"].append(filename)
                new_rows["labels"].append(no_class_idx)

            # This helps make sure stuff isn't loaded into memory
            # Hopefully
            file_ds = Dataset.from_dict(new_rows).cast_column(
                "audio", SubAudio()
            )
            new_rows = get_empty_dict()
            _datasets.append(file_ds)
    return concatenate_datasets(_datasets)


def one_hot_encode(row: dict, classes: list):
    """One hot Encodes a list of labels.

    Args:
        row (dict): row of data in a dataset containing a labels column
        classes: a list of classes
    """
    one_hot = np.zeros(len(classes))
    one_hot[row["labels"]] = 1
    row["labels"] = np.array(one_hot, dtype=int)
    return row


# output_folder is there as legacy
def raw_audio_extractor(
    audio_parent_folder: str = "",
    class_list=None,
    chunk_duration=-1,
    output_folder="",  # pylint: disable=unused-argument
):
    """Extracts raw, unlabeled data in the buowset format into an AudioDataset.

    Args:
        audio_parent_folder (str): Path to the parent folder for all audio data
            Note its assumed the audio filepath
            in the csv is relative to parent_path
        sr (int): Sample Rate of the audio files Default: 32_000

    Returns:
        (AudioDataset): See dataset.py, AudioDatasets are consider
        the universal dataset for the training pipeline.
    """
    dataset = get_array_chunks_from_memory(
        parent_folder=audio_parent_folder,
        chunk_length_sec=chunk_duration,
    )

    # # # # Convert to a uniform one_hot encoding for classes
    dataset = dataset.class_encode_column("labels")
    multilabel_class_label = Sequence(ClassLabel(names=class_list))
    dataset = dataset.map(
        lambda row: one_hot_encode(row, class_list)).cast_column(
        "labels", multilabel_class_label
    )

    ds = AudioDataset(
        DatasetDict({"train": dataset, "valid": dataset, "test": dataset})
    )
    return ds
