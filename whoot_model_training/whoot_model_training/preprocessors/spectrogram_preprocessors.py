"""Defines preprocessors for creating spectrograms.

Pulled from pyha_analyzer/preprocessors/spectogram_preprocessors.py
"""

from dataclasses import dataclass

import librosa
import numpy as np
from torchvision import transforms

from pyha_analyzer.preprocessors import PreProcessorBase


@dataclass
class SpectrogramParams:
    """Dataclass for spectrogram Parameters.

    n_fft: (int) number of fft bins
    hop_length (int) skip count
    power: (float) usually 2
    n_mels: (int) number of mel bins
    """

    n_fft: int = 2048
    hop_length: int = 256
    power: float = 2.0
    n_mels: int = 256


@dataclass
class Augmentations:
    """Dataclass for the augmentations of the model.

    audio (list[dict]): per item key name of augmentation,
        value is the augmentation
    spectrogram (list[dict]): same idea but augmentations
        applied onto spectrograms
    """

    audio = None
    spectrogram = None


class BuowMelSpectrogramPreprocessors(PreProcessorBase):
    """Preprocessor for processing audio into spectrograms.

    Particularly for the buow dataset
    """

    def __init__(
        self,
        duration=5,
        augments: Augmentations = Augmentations(),
        spectrogram_params: SpectrogramParams = SpectrogramParams(),
    ):
        """Defines a BuowMelSpectrogramPreprocessors.

        Args:
            duration (float): length of chunk of data to train on
            augments (Augmentations): An augmentation to apply to waveforms
            spectrogram_params (SpectrogramParams):
                config for spectrogram generation
        """
        self.duration = duration
        self.augments = augments

        # Below parameter defaults from https://arxiv.org/pdf/2403.10380 pg 25
        self.n_fft = spectrogram_params.n_fft
        self.hop_length = spectrogram_params.hop_length
        self.power = spectrogram_params.power
        self.n_mels = spectrogram_params.n_mels
        self.spectrogram_params = spectrogram_params

        super().__init__(name="MelSpectrogramPreprocessor")

    def __call__(self, batch):
        """Process a batch of data from an AudioDataset."""
        new_audio = []
        new_labels = []
        for item_idx in range(len(batch["audio"])):
            label = batch["labels"][item_idx]
            y, sr = (
                batch["audio"][item_idx]["array"],
                batch["audio"][item_idx]["sampling_rate"],
            )
            start = 0

            # Handle out of bound issues
            end_sr = int(start * sr) + int(sr * self.duration)
            if y.shape[-1] <= end_sr:
                y = np.pad(y, end_sr - y.shape[-1])

            # Audio Based Augmentations
            if self.augments.audio is not None:
                y, label = self.augments.audio(y, sr, label)

            pillow_transforms = transforms.ToPILImage()

            S = librosa.feature.melspectrogram(
                y=y[int(start * sr):end_sr],
                sr=sr,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                power=self.power,
                n_mels=self.n_mels,
            )
            pcen_S = librosa.pcen(S * (2**31))

            mels = (
                np.array(
                    pillow_transforms(
                        pcen_S
                    ),
                    np.float32,
                )[np.newaxis, ::]
                / 255
            )

            if self.augments.spectrogram is not None:
                mels = self.augments.spectrogram(mels)

            new_audio.append(mels)
            new_labels.append(label)

        batch["audio"] = np.concatenate(new_audio)
        batch["labels"] = np.array(new_labels, dtype=np.float32)

        return batch

    def get_augmentations(self):
        """Returns a list of augmentations.

        Perhaps for logging purposes

        Returns:
            (list) all the augmentations
        """
        return self.augments

    def __repr__(self):
        """Use representation to describe the augmentations.

        Returns:
            (str) all information about this preprocessor
        """
        return f"""{self.name}
                Augmentations: {self.augments}
                MelSpectrogram: {self.spectrogram_params}
            """


class PCENMelSpectrogramPreprocessors(BuowMelSpectrogramPreprocessors):
    """A preprocessor for PCEN based spectograms.

    Otherwise follows the same system as BuowMelSpectrogramPreprocessors
    """

    def __call__(self, batch):
        """Process a batch of data from an AudioDataset."""
        new_audio = []
        new_labels = []
        for item_idx in range(len(batch["audio"])):
            label = batch["labels"][item_idx]
            y, sr = (
                batch["audio"][item_idx]["array"],
                batch["audio"][item_idx]["sampling_rate"],
            )
            start = 0

            # Handle out of bound issues
            end_sr = int(start * sr) + int(sr * self.duration)
            if y.shape[-1] <= end_sr:
                y = np.pad(y, end_sr - y.shape[-1])

            # Audio Based Augmentations
            if self.augments.audio is not None:
                y, label = self.augments.audio(y, sr, label)

            pillow_transforms = transforms.ToPILImage()

            spec = librosa.feature.melspectrogram(
                y=y[int(start * sr):end_sr],
                sr=sr,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                power=self.power,
                n_mels=self.n_mels,
            )

            pcen = librosa.pcen(spec * (2**31))

            mels = (
                np.array(
                    pillow_transforms(
                        pcen
                    ),
                    np.float32,
                )[np.newaxis, ::]
                / 255
            )

            if self.augments.spectrogram is not None:
                mels = self.augments.spectrogram(mels)

            new_audio.append(mels)
            new_labels.append(label)

        batch["audio"] = new_audio
        batch["labels"] = np.array(new_labels, dtype=np.float32)

        return batch
