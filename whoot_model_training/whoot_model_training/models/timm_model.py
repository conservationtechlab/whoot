"""Wrapper around the timms model zoo 

    See https://timm.fast.ai/

    Timm model zoo good for computer vision models
    Like CNNs, which are useful for spectrograms 

    Great repo for models, but currently using this for demoing pipeline
"""

import timm
from torch import nn, Tensor
import numpy as np

from .model import Model, ModelInput, ModelOutput, has_required_inputs


class TimmInputs(ModelInput):
    """Input for TimmModel's

    Spefifies TimmModels needs labels and spectrograms that are Tensors
    """
    def __init__(self, labels, waveform=None, spectrogram=None, device="cpu"):
        # # Can use inputs to verify correct shape for upstream model
        # assert spectrogram.shape[1:] == (1, 100, 100)
        super().__init__(labels, waveform, spectrogram)
        self.labels = Tensor(np.array(labels))
        self.spectrogram = Tensor(np.array(spectrogram))


class TimmModel(nn.Module, Model):
    """Model that uses a timm's model as its backbone with a linear layer for classification
    """

    def __init__(
        self,
        timm_model="resnet34",
        pretrained=True,
        in_chans=1,
        num_classes=6,
        loss=None,
    ):
        """
        kwargs:
            timm_model (str): name of model backbone from timms to use, Default: "resnet34" 
            pretrained (bool): use a pretrained model from timms, Default: True
            in_chans (int): number of channels of audio: Default: 1
            num_classes (int): number of classes in the dataset: Default 6
            loss (any): custom loss function Default: BCEWithLogitsLoss
        """
        super().__init__()
        self.input_format = TimmInputs
        self.output_format = ModelOutput

        assert num_classes > 0

        self.backbone = timm.create_model(
            timm_model, pretrained=pretrained, in_chans=in_chans
        )
        # Unsure if 1000 is default for all models. Need to check this
        self.linear = nn.Linear(1000, num_classes)

        # Models might need different losses during training!
        if loss is not None:
            self.loss = loss
        else:
            self.loss = nn.BCEWithLogitsLoss()

    @has_required_inputs()  # data: TimmInputs
    def forward(self, spectrogram, labels=None) -> ModelOutput:
        # print(len(spectrogram)) # batch size
        embedd = self.backbone(spectrogram)
        logits = self.linear(embedd)
        loss = self.loss(logits, labels)
        out = ModelOutput(logits=logits, embeddings=embedd, loss=loss, labels=labels)
        return out
