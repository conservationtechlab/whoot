""" A collection of online preprocessors

During training online preprocessors convert data
into data ready to be given to a model

In traditional pytorch world, this would be like
the __get_item__ function of a dataset
"""

from .default_preprocessor import (
    SpectrogramModelInputPreprocessors
)
from .spectrogram_preprocessors import (
    BuowMelSpectrogramPreprocessors
)

__all__ = [
    "SpectrogramModelInputPreprocessors",
    "BuowMelSpectrogramPreprocessors"
]
