"""A collection of online preprocessors.

During training online preprocessors convert data
into data ready to be given to a model

In traditional pytorch world, this would be like
the __get_item__ function of a dataset
"""

from .base_preprocessor import (
    MelModelInputPreprocessor, WaveformInputPreprocessor
)
from .spectrogram_preprocessors import (
    BuowMelSpectrogramPreprocessors
)

__all__ = [
    "MelModelInputPreprocessor",
    "BuowMelSpectrogramPreprocessors",
    "WaveformInputPreprocessor"
]
