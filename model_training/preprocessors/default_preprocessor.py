from pyha_analyzer.preprocessors import MelSpectrogramPreprocessors
from models.model import ModelInput


"""_summary_

Returns:
    _type_: _description_
"""


class SpectrogramModelInputPreprocessors(MelSpectrogramPreprocessors):
    def __init__(
        self,
        ModelInput: ModelInput,
        duration=5,
        augment=None,
        spectrogram_augments=None,
        class_list=...,
        n_fft=2048,
        hop_length=256,
        power=2,
        n_mels=256,
        dataset_ref=None,
    ):
        super().__init__(
            duration,
            augment,
            spectrogram_augments,
            class_list,
            n_fft,
            hop_length,
            power,
            n_mels,
            dataset_ref,
        )
        self.ModelInput = ModelInput

    def __call__(self, batch):
        batch = super().__call__(batch)
        return self.ModelInput(labels=batch["labels"], spectrogram=batch["audio"])
