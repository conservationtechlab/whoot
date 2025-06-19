import timm
from torch import nn, Tensor

from .model import Model, ModelInput, ModelOutput, has_required_inputs

"""
    Wrapper around the timms model zoo 

    See https://timm.fast.ai/

    Timm model zoo good for computer vision models
    Like CNNs, which are useful for spectrograms 

    Great repo for models, but currently using this for demoing pipeline
"""


class TimmInputs(ModelInput):
    def __init__(self, labels, waveform=None, spectrogram=None, device="cpu"):
        # # Can use inputs to verify correct shape for upstream model
        # assert spectrogram.shape[1:] == (1, 100, 100)
        super().__init__(labels, waveform, spectrogram)
        self.labels = Tensor(labels)
        self.spectrogram = Tensor(spectrogram)


class TimmModel(nn.Module, Model):
    def __init__(
        self,
        timm_model="resnet34",
        pretrained=True,
        in_chans=1,
        num_classes=6,
        loss=None,
    ):
        super().__init__()
        self.input_format = TimmInputs
        self.output_format = ModelOutput

        assert num_classes > 0

        self.backbone = timm.create_model(
            timm_model, pretrained=pretrained, in_chans=in_chans
        )
        # Unsure if 1000 is default for all models. Need to check this
        self.linear = nn.Linear(1000, num_classes)

        # Models might need diffrent losses during training!
        if loss is not None:
            self.loss = loss
        else:
            self.loss = nn.BCEWithLogitsLoss()

    @has_required_inputs() #data: TimmInputs TODO FIX
    def forward(self, labels=None, spectrogram=None) -> ModelOutput:
        embedd = self.backbone(spectrogram)
        logits = self.linear(embedd)
        loss = self.loss(logits, labels)

        return ModelOutput(logits=logits, embeddings=embedd, loss=loss, labels=labels)
