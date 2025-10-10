"""Default Class for Preprocessing the data.

The dataset is one thing, what we feed into the models is another
Models may require spectrograms, waveforms, etc

Not to mention any online augmentation we want to do

The preprocessor class defines a function to preprocess our data during
training

The default preprocessor allows for many types of preprocessors to run,
but it forces the output to fit the ModelInput class structure.
see `whoot_model_training/models/model.py` for more info.
"""
# pylint: disable=too-few-public-methods

from pyha_analyzer.preprocessors import PreProcessorBase

from .spectrogram_preprocessors import (
    BuowMelSpectrogramPreprocessors,
    SpectrogramParams,
    Augmentations
)
from ..models.model import ModelInput

from .waveform_preprocessors import (
    WaveformPreprocessors
)


class SpectrogramModelInPreprocessors(PreProcessorBase):
    """Defines a preprocessor that after formatting the audio.

    Passes a spectrogram into a ModelInput object.
    """
    def __init__(
        self,
        spec_preprocessor: PreProcessorBase,
        model_input: ModelInput,
    ):
        """Wrapper to get the raw spectrogram output of spec_preprocessor.

        and format it neatly into a model_input

        Args:
            spec_preprocessor (PreProcessorBase): a preprocessor that
                creates spectrograms
            model_input (ModelInput): How the model like input data formatted
        """
        self.spec_preprocessor = spec_preprocessor
        self.model_input = model_input
        super().__init__(name="SpectrogramModelInPreprocessors")

    def __call__(self, batch: dict) -> ModelInput:
        """Processes a batch of AudioDataset rows.

        For this specific preprocessor, it creates a spectrogram then
        Formats the data as a ModelInput
        """
        batch = self.spec_preprocessor(batch)
        return self.model_input(
            labels=batch["labels"],
            spectrogram=batch["audio"]
        )


class MelModelInputPreprocessor(SpectrogramModelInPreprocessors):
    """Demo of how SpectrogramModelInPreprocessors works.

    Uses a kind of Spectrogram Preprocessor, BuowMelSpectrogramPreprocessors

    This was created in part because legacy implementation of
    SpectrogramModelInputPreprocessors had these parameters and subclassed
    BuowMelSpectrogramPreprocessors. This class replicates the
    format of the old SpectrogramModelInputPreprocessors
    class with the new functionality
    """
    def __init__(
        self,
        model_input: ModelInput,
        duration=5,
        augments: Augmentations = Augmentations(),
        spectrogram_params: SpectrogramParams = SpectrogramParams(),
    ):
        """Creates a Online preprocessor for MelSpectrograms Based Models.

        Formats input into spefific ModelInput format.

        Args:
            model_input (ModelInput): How the model like input data formatted
            duration (int): Length in seconds of input
            augments (dict): contains two keys: audio,
                spectrogram each defining
                a dict of augmentation names and augmentations to run
            spectrogram_params (SpectrogramParams):
                has the following parameters:
                    class_list (list): the classes we are
                        working with one-hot-encoding
                    n_fft (int): number of ffts
                    hop_length (int): hop length
                    power (int): power, defined by librosa
                    n_mels (int): number of mels for a melspectrogram
                    dataset_ref (AudioDataset): a
                        external ref to an AudioDataset
        """
        spec_preprocessor = BuowMelSpectrogramPreprocessors(
            duration=duration,
            augments=augments,
            spectrogram_params=spectrogram_params
        )
        super().__init__(spec_preprocessor, model_input)


class WaveformInputPreprocessor(SpectrogramModelInPreprocessors):
    """Demo of how SpectrogramModelInPreprocessors works.

    Uses a kind of Spectrogram Preprocessor, BuowMelSpectrogramPreprocessors

    This was created in part because legacy implementation of
    SpectrogramModelInputPreprocessors had these parameters and subclassed
    BuowMelSpectrogramPreprocessors. This class replicates the
    format of the old SpectrogramModelInputPreprocessors
    class with the new functionality
    """
    def __init__(
        self,
        model_input: ModelInput,
        duration=5,
        augments: Augmentations = Augmentations(),
    ):
        """Creates a Online preprocessor for MelSpectrograms Based Models.

        Formats input into spefific ModelInput format.

        Args:
            model_input (ModelInput): How the model like input data formatted
            duration (int): Length in seconds of input
            augments (dict): contains two keys: audio,
                spectrogram each defining
                a dict of augmentation names and augmentations to run
            spectrogram_params (SpectrogramParams):
                has the following parameters:
                    class_list (list): the classes we are
                        working with one-hot-encoding
                    dataset_ref (AudioDataset): a
                        external ref to an AudioDataset
        """
        wav_preprocessor = WaveformPreprocessors(
            duration=duration,
            augments=augments,
        )
        super().__init__(wav_preprocessor, model_input)

    def __call__(self, batch: dict) -> ModelInput:
        """Processes a batch of AudioDataset rows.

        For this specific preprocessor, it creates a spectrogram then
        Formats the data as a ModelInput
        """
        batch = self.spec_preprocessor(batch)
        return self.model_input(
            labels=batch["labels"],
            waveform=batch["audio"]
        )





