"""Build a few_shot_learning classifier.

Inspired by the work of
Jacuzzi, G., Olden, J.D., 2025.
Few-shot transfer learning enables robust acoustic
monitoring of wildlife communities at the landscape scale.
Ecological Informatics 90, 103294.
doi.org/10.1016/j.ecoinf.2025.103294

These models convert thier input into an embedding from a large audio model and
do processing on top of that embedding
"""

# from torch import nn, Tensor
# from perch_hoplite.zoo import model_configs
# from .model import Model, ModelInput, ModelOutput, has_required_inputs

from transformers import PretrainedConfig
from .model import ModelInput


class EmbeddingModel():
    """Wrapper for models which are only intended for embeddings."""
    def embed(self):
        """Get embedding."""
        raise NotImplementedError()

    def get_k_neighbors(self):
        """Get k nearest neighbors."""
        raise NotImplementedError()


class EmbeddingInput(ModelInput):
    """Wrapper for ModelInputs that are embeddings."""
    model = EmbeddingModel()
    embedding_size = 0

    def __init__(
        self,
        labels,
        waveform=None,
        spectrogram=None
    ):
        """.

        Args:
            labels: label
            waveform: np array of sound
            spectrogram: 2d array representing sound
        """
        super().__init__(labels, waveform, spectrogram)

        # I keep getting this linting error
        # But there is not too many function args here
        # pylint: disable=too-many-function-args
        self["embedding"] = self.model.embed(waveform)


# Global variable fore PerchEmbeddings
PERCH_MODEL = None

# class PerchEmbeddings(EmbeddingModel):
#     """Wrapper for getting embeddings from perch."""

#     # Warning, was running into issues with memory here
#     # Early attempts recreated model
#     # Hoping using global var only loads it in once
#     if perch_model is None:
#         perch_model = model_configs.load_model_by_name('perch_8')

#     model = perch_model

#     def embed(self, embeddings):
#         """Return embeddings."""
#         return embeddings


# class PerchEmbeddingInput(EmbeddingInput):
#     """Wrapper for an input into a larger model from perch."""
#     model = PerchEmbeddings()
#     embedding_size = 1280


class FewShotModelConfig(PretrainedConfig):
    """Config for Timm Model Zoo Models!"""
    def __init__(
        self,
        num_classes=200,
        **kwargs
    ):
        """Creates Config.

        Args:
            num_classes: how many species we want to detect
        """
        self.num_classes = num_classes
        super().__init__(**kwargs)


# class PerchFewShotModel(Model, nn.Module):
#     """Perch model intergration with pytorch."""
#     def __init__(
#         self,
#         config: FewShotModelConfig
#     ):
#         """Init for TimmModel.

#         kwargs:
#             timm_model (str): name of model backbone from timms to use,
#                 Default: "resnet34"
#             pretrained (bool): use a pretrained model from timms,
#                 Default: True
#             in_chans (int): number of channels of audio: Default: 1
#             num_classes (int): number of classes in the dataset: Default 6
#             loss (any): custom loss function Default: BCEWithLogitsLoss
#         """
#         super().__init__()

#         self.input_format = PerchEmbeddingInput
#         self.output_format = ModelOutput

#         self.config = config
#         assert config.num_classes > 0

#         self.linear = nn.Linear(
#             self.input_format.embedding_size,
#             config.num_classes
#         )

#         self.loss = nn.BCEWithLogitsLoss()

#     @has_required_inputs()
#     def forward(self, x: PerchEmbeddingInput):
#         """Run model over x!"""
#         # Use perch to create embeddings
#         embeddings = Tensor(
#             x.model.model.embed(x["waveform"].cpu()).embeddings
#         ).to(x["waveform"].device)

#         logits = self.linear(embeddings).squeeze(1)
#         loss = self.loss(logits, x["labels"])

#         return ModelOutput(
#             logits=logits,
#             embeddings=embeddings,
#             loss=loss,
#             labels=x["labels"]
#         )
