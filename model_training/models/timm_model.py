from .model import Model, ModelInput, ModelOutput
import timm 
from torch import nn

"""
    Wrapper around the timms model zoo 

    See https://timm.fast.ai/

    Timm model zoo good for computer vision models
    Like CNNs, which are useful for spectrograms 

    Great repo for models, but currently using this for demoing pipeline
"""
class TimmInputs(ModelInput):
    def __init__(self, waveform = None, spectrogram = None):
        # Can use inputs to verify correct shape for upstream model
        assert spectrogram.shape[1:] == (1, 100, 100)
        super().__init__(None, spectrogram)


class TimmModel(nn.Module, Model):
    def __init__(self, timm_model='resnet34', pretrained=True, in_chans=1, num_classes=6, loss=None):
        assert num_classes > 0

        self.backbone = timm.create_model(timm_model, pretrained=pretrained, in_chans=in_chans)
        # Unsure if 1000 is default for all models. Need to check this
        self.linear = nn.Linear(1000, num_classes)

        # Models might need diffrent losses during training!
        if loss is not None:
            self.loss = loss
        else:
            self.loss = nn.BCEWithLogitsLoss()
    
    def forward(self, x: TimmInputs) -> ModelOutput:
        embedd = self.backbone(x)
        logits = self.linear(embedd)
        loss = self.loss(logits)

        return ModelOutput(
            logits=logits,
            embeddings=embedd,
            loss=loss
        )
