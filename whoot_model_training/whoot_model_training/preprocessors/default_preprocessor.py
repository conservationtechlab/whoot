"""Default Class for Preprocessing the data

The dataset is one thing, what we feed into the models is another
Models may require spectrograms, waveforms, etc

Not to mention any online augmentation we want to do

The preprocessor class defines a function to preprocess our data during
training

The default preprocessor allows for many types of preprocessors to run,
but it forces the output to fit the ModelInput class structure.
see `whoot_model_training/models/model.py` for more info.
"""


from .spectrogram_preprocessors import (
    BuowMelSpectrogramPreprocessors, SpectrogramParams
)
from ..models.model import ModelInput


class SpectrogramModelInputPreprocessors(BuowMelSpectrogramPreprocessors):
    """ Defines a preprocessed that after formatting the audio
    passes a spectrogram into a ModelInput object.
    """
    def __init__(
        self,
        model_input: ModelInput,
        duration=5,
        augments: dict = {"audio":None, "spectrogram":None},
        spectrogram_params: SpectrogramParams = SpectrogramParams(),
    ):
        """ Creates a Online preprocessor for MelSpectrograms Based Models

        Formats input into spefific ModelInput format.

        Args:
            ModelInput (ModelInput): How the model like input data formatted
            Duration (int): Length in seconds of input
            augment (dict): contains two keys: audio, spectrogram each defining
                a dict of augmentation names and augmentations to run
            class_list (list): the classes we are working with one-hot-encoding
            n_fft (int): number of ffts
            hop_length (int): hop length
            power (int): power, defined by librosa
            n_mels (int): number of mels for a melspectrogram
            dataset_ref (AudioDataset): a external ref to an AudioDataset
        """
        super().__init__(
            duration,
            augments,
            spectrogram_params
        )
        self.model_input = model_input

    def __call__(self, batch: dict) -> ModelInput:
        """Processes a batch of AudioDataset rows

        For this specific preprocessor, it creates a spectrogram then
        Formats the data as a ModelInput
        """
        batch = super().__call__(batch)
        return self.model_input(
            labels=batch["labels"],
            spectrogram=batch["audio"]
        )
