"""Defines a default preprocessor class.

Now this allows for defining a set of common audio loading utilities.
"""
from dataclasses import dataclass
import librosa
import numpy as np
from pyha_analyzer.preprocessors import PreProcessorBase


@dataclass
class Augmentations():
    """Dataclass for the augmentations of the model.

    audio (list[dict]): per item key name of augmentation,
        value is the augmentation
    spectrogram (list[dict]): same idea but augmentations
        applied onto spectrograms
    """
    audio = None
    spectrogram = None


class DefaultPreprocessor(PreProcessorBase):
    """Default Preprocessor class."""
    def __init__(self, name, duration, sr, *args, **kwargs):
        """Initializes the DefaultPreprocessor.

        Args:
            name (str): name of preprocessor for logging
            duration (float): max length in seconds of audio chunk
            sr (int/None): sample rate to standardize audio to
        """
        super().__init__(name, *args, **kwargs)
        self.duration = duration
        self.sr = sr

    def load_audio(self, batch, item_idx):
        """Load audio from either array or path.

        Args:
            batch (dict):  AudioDataset batch
            item_idx (int): Processing an item in batch
        Returns:
            y (np.ndarray): audio array loaded
            sr (int): sample rate of audio
        """
        try:
            if len(batch["audio"][item_idx]["array"]) > 10:
                y = batch["audio"][item_idx]["array"]
                sr = batch["audio"][item_idx]["sampling_rate"]
            else:
                if librosa.get_duration(
                    path=batch["audio"][item_idx]["path"]
                ) > 2 * 60:
                    raise IOError("File too long to process")

                y, sr = librosa.load(
                    path=batch["audio"][item_idx]["path"],
                    sr=self.sr
                )

        except IOError as e:
            y = np.zeros(self.sr * 5)
            sr = self.sr
            print("File Likely is corrupted, moving on", e)
            raise IOError from e

        return y, sr

    def augment_audio(
        self,
        y: np.ndarray,
        sr: int,
        start: float,
        label: str,
        augments: Augmentations
    ):
        """Placeholder for audio augmentations.

        Args:
            y: audio array
            sr: sample rate
            label: label associated with audio
            start: starting point in seconds to crop audio
            augments: augmentations to apply
        """
        # Handle out of bound issues
        end_sr = int(start * sr) + int(sr * self.duration)
        if y.shape[-1] <= end_sr:
            y = np.pad(y, end_sr - y.shape[-1])

        # Audio Based Augmentations
        if augments.audio is not None:
            y, label = augments.audio(y, sr, label)

        new_y = y[int(start * sr):end_sr]
        if new_y.shape[-1] < int(sr * self.duration):
            raise IOError("Audio too short after augmentation")

        return new_y, label
