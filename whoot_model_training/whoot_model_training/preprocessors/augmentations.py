"""Contains various data augementation techinques for bioacoustics
Notes: relies heavily on the audiomentions library

Basically combine augmentations with ComposeAudioLabel

For clarity, put augmentations imports here

For Devs:
To create a new augmentation, create a AudioLabelPreprocessor
"""
from pyha_analyzer.preprocessors.augmentations import ComposeAudioLabel, MixItUp, AudioLabelPreprocessor
from audiomentations import Gain, PolarityInversion

