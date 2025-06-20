"""
Pulled from pyha_analyzer/preprocessors/spectogram_preprocessors.py
"""

import librosa
import numpy as np
import torchvision.transforms as transforms

from pyha_analyzer.preprocessors import PreProcessorBase


class BuowMelSpectrogramPreprocessors(PreProcessorBase):
    def __init__(
        self,
        duration=5,
        augment=None,
        spectrogram_augments=None,
        class_list=[],
        n_fft=2048,
        hop_length=256,
        power=2.0,
        n_mels=256,
        dataset_ref=None,
    ):
        self.duration = duration
        self.augment = augment
        self.spectrogram_augments = spectrogram_augments

        # Below parameter defaults from https://arxiv.org/pdf/2403.10380 pg 25
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.power = power
        self.n_mels = n_mels

        super().__init__(name="MelSpectrogramPreprocessor")

    def __call__(self, batch):
        new_audio = []
        new_labels = []
        for item_idx in range(len(batch["audio"])):
            label = batch["labels"][item_idx]
            y, sr = librosa.load(path=batch["audio"][item_idx]["path"])
            start = 0

            # Handle out of bound issues
            end_sr = int(start * sr) + int(sr * self.duration)
            if y.shape[-1] <= end_sr:
                y = np.pad(y, end_sr - y.shape[-1])

            # Audio Based Augmentations
            if self.augment is not None:
                y, label = self.augment(y, sr, label)

            pillow_transforms = transforms.ToPILImage()

            mels = (
                np.array(
                    pillow_transforms(
                        librosa.feature.melspectrogram(
                            y=y[int(start * sr) : end_sr],
                            sr=sr,
                            n_fft=self.n_fft,
                            hop_length=self.hop_length,
                            power=self.power,
                            n_mels=self.n_mels,
                        )
                    ),
                    np.float32,
                )[np.newaxis, ::]
                / 255
            )

            if self.spectrogram_augments is not None:
                mels = self.spectrogram_augments(mels)

            new_audio.append(mels)
            new_labels.append(label)

        batch["audio"] = new_audio
        batch["labels"] = np.array(new_labels, dtype=np.float32)

        return batch
