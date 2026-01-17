"""Defines preprocessors for creating spectrograms.

Pulled from pyha_analyzer/preprocessors/spectogram_preprocessors.py
"""

import numpy as np
from .default_preprocessor import DefaultPreprocessor, Augmentations

# @dataclass
# class WaveformParams:
#     """Dataclass for spectrogram Parameters.

#     n_fft: (int) number of fft bins
#     hop_length (int) skip count
#     power: (float) usually 2
#     n_mels: (int) number of mel bins
#     """
#     n_fft: int = 2048
#     hop_length: int = 256
#     power: float = 2.0
#     n_mels: int = 256


class WaveformPreprocessors(DefaultPreprocessor):
    """Preprocessor for processing audio into spectrograms.

    Particularly for the buow dataset
    """

    def __init__(
        self,
        duration=5,
        sr=None,
        augments: Augmentations = Augmentations(),
    ):
        """Defines a BuowMelSpectrogramPreprocessors.

        Args:
            duration (float): length of chunk of data to train on
            augments (Augmentations): An augmentation to apply to waveforms
            sr (int/None): sample rate of audio to standize,
                defaults to use file sr
            spectrogram_params (SpectrogramParams):
                config for spectrogram generation
        """
        self.duration = duration
        self.augments = augments
        self.sr = sr

        # # Below parameter defaults from
        # # https://arxiv.org/pdf/2403.10380 pg 25
        # self.n_fft = spectrogram_params.n_fft
        # self.hop_length = spectrogram_params.hop_length
        # self.power = spectrogram_params.power
        # self.n_mels = spectrogram_params.n_mels
        # self.spectrogram_params = spectrogram_params

        super().__init__(
            name="MelSpectrogramPreprocessor",
            duration=duration,
            sr=self.sr)

    def __call__(self, batch):
        """Process a batch of data from an AudioDataset."""
        new_audio = []
        new_labels = []
        for item_idx in range(len(batch["audio"])):
            label = batch["labels"][item_idx]

            y, sr = self.load_audio(batch, item_idx)

            start = np.random.uniform(0, len(y)/sr - self.duration)

            y, label = self.augment_audio(y, sr, start, label, self.augments)

            new_audio.append(y)
            new_labels.append(label)

        batch["audio"] = new_audio
        batch["labels"] = np.array(new_labels, dtype=np.float32)
        # print(len(batch["audio"]),  len(batch["labels"]))

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
        return (
            f"""{self.name}
                Augmentations: {self.augments}
            """
        )
