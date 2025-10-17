"""Wrapper around the hugging face model api!"""

from transformers import AutoFeatureExtractor, AutoModel
from torch import nn
import torch
from contextlib import nullcontext
from transformers import PretrainedConfig

from .model import Model, ModelInput, ModelOutput, has_required_inputs


class HFInput():
    """Input for Hugging Face Models.

    Specifies TimmModels needs labels and spectrograms that are Tensors
    """
    def __init__(self,
                 labels=None,
                 spectrogram=None,
                 waveform=None,
                 extractor_path="DBD-research-group/Bird-MAE-Base"):
        """Creates TimmInputs.

        Args:
            labels: the data's label for this batch
            spectrogram: Legacy
            waveform: Legacy
            extractor_path: Path to hugging face preprocessor
        """
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            extractor_path,
            trust_remote_code=True)
        # TODO MAKE HFINPUT WORK WITH ITSELF

    def __call__(self, labels, spectrogram=None,  waveform=None):
        """Create some fake ModelInputs for HFModels.

        Slightly diffrent API for HFInput, when creating a input
        Use the preprocessor from hugging face.
        """
        mel_spectrogram = self.feature_extractor(waveform)
        return ModelInput(labels, waveform=None, spectrogram=mel_spectrogram)


class HFModelConfig(PretrainedConfig):
    """Config for Timm Model Zoo Models!"""
    def __init__(
        self,
        path: str = "DBD-research-group/Bird-MAE-Huge",
        num_classes: int = 6,
        embeddings_size: int = 1280,
        freeze_backbone: bool = True,
        **kwargs
    ):
        """Creates Config.

        Args:
            path (str): url to pull from hf model zoo
            num_classes (int): number of classes in dataset, for cls
            embeddings_size (int): size of output of model
            freeze_backbone (bool): freeze the backbone of a model
        """
        self.path = path
        self.num_classes = num_classes
        self.embeddings_size = embeddings_size
        self.freeze_backbone = freeze_backbone
        super().__init__(**kwargs)


class HFModel(Model, nn.Module):
    """Model that uses a timm's model."""
    config_class = HFModelConfig

    def __init__(
        self,
        config: HFModelConfig
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
        self.input_format = ModelInput
        self.output_format = ModelOutput
        self.config = config
        assert config.num_classes > 0

        # Deep learning CNN backbone
        self.backbone = AutoModel.from_pretrained(
            config.path,
            trust_remote_code=True
        )

        # Unsure if 1000 is default for all timm models. Need to check this
        self.linear = nn.Linear(config.embeddings_size, config.num_classes)

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
    def forward(self, x: HFInput) -> ModelOutput:
        """Model forward function.

        Args:
            x: (TimmInputs): The specific input format for Timm Models

        Returns
            (ModelOutput): The model output (logits),
            latent space representations (embeddings), loss and labels.
        """
        with torch.no_grad() if self.config.freeze_backbone else nullcontext():
            embed = self.backbone(
                x.spectrogram.to(self.device)
            ).last_hidden_state
        logits = self.linear(embed)
        loss = self.loss(logits, x.labels)

        return ModelOutput(
            logits=logits,
            embeddings=embed,
            loss=loss,
            labels=x.labels
        )
