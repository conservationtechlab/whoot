"""Wrapper around the timms model zoo!

See https://timm.fast.ai/

Timm model zoo good for computer vision models
Like CNNs, which are useful for spectrograms

Great repo for models, but currently using this for demoing pipeline
"""

import timm
from torch import nn
from transformers import PretrainedConfig

from .model import Model, ModelInput, ModelOutput, has_required_inputs


class TimmInputs(ModelInput):
    """Input for TimmModels.

    Specifies TimmModels needs labels and spectrograms that are Tensors
    """
    def __init__(self, labels, spectrogram=None):
        """Creates TimmInputs.

        Args:
            labels: the data's label for this batch
            spectrogram: audio's spectrogram
            waveform: Optional, audio waveform
        """
        # # Can use inputs to verify correct shape for upstream model
        # assert spectrogram.shape[1:] == (1, 100, 100)
        super().__init__(labels, waveform=None, spectrogram=spectrogram)
        self.labels = labels
        self.spectrogram = spectrogram


class TimmModelConfig(PretrainedConfig):
    """Config for Timm Model Zoo Models!"""
    def __init__(
        self,
        timm_model="resnet34",
        pretrained=True,
        in_chans=1,
        num_classes=6,
        **kwargs
    ):
        """Creates Config.

        Args:
            timm_model (str): name of a model in timm model zoo
            pretrained (bool): use pretrain weights from timms
            in_chans (int): channels in audio, mono is 1
            num_classes (int): number of classes in dataset, for cls
        """
        self.timm_model = timm_model
        self.pretrained = pretrained
        self.in_chans = in_chans
        self.num_classes = num_classes
        super().__init__(**kwargs)


class TimmModel(Model, nn.Module):
    """Model that uses a timm's model."""
    config_class = TimmModelConfig

    def __init__(
        self,
        config: TimmModelConfig
    ):
        """Init for TimmModel.

        kwargs:
            timm_model (str): name of model backbone from timms to use,
                Default: "resnet34"
            pretrained (bool): use a pretrained model from timms, Default: True
            in_chans (int): number of channels of audio: Default: 1
            num_classes (int): number of classes in the dataset: Default 6
            loss (any): custom loss function Default: BCEWithLogitsLoss
        """
        super().__init__()
        self.input_format = TimmInputs
        self.output_format = ModelOutput
        self.config = config
        assert config.num_classes > 0

        # Deep learning CNN backbone
        self.backbone = timm.create_model(
            config.timm_model,
            pretrained=config.pretrained,
            in_chans=config.in_chans
        )

        # Unsure if 1000 is default for all timm models. Need to check this
        self.linear = nn.Linear(1000, config.num_classes)

        # different losses if you want to train for different problems
        # BCEWithLogitsLoss is default as for Bioacoustics, the problem tends
        # multilabel!
        # the probability of class A occurring doesn't
        # change the probability of Class B
        # Many individuals can make calls at the same time!
        self.loss = nn.BCEWithLogitsLoss()

    def set_custom_loss(self, loss_fn):
        """Set a different loss function.

        For cases where we don't want BCEWithLogitsLoss

        Args:
            loss_fn: Function to compute loss, ideally in pytorch
        """
        self.loss = loss_fn

    @has_required_inputs()
    def forward(self, x: TimmInputs) -> ModelOutput:
        """Model forward function.

        Args:
            x: (TimmInputs): The specific input format for Timm Models

        Returns
            (ModelOutput): The model output (logits),
            latent space representations (embeddings), loss and labels.
        """
        embed = self.backbone(x.spectrogram)
        logits = self.linear(embed)
        loss = self.loss(logits, x.labels)

        return ModelOutput(
            logits=logits,
            embeddings=embed,
            loss=loss,
            labels=x.labels
        )
