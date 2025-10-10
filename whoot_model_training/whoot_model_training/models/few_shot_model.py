"""Build a few_shot_learning classifier.

Inspired by the work of 
Jacuzzi, G., Olden, J.D., 2025. Few-shot transfer learning enables robust acoustic 
monitoring of wildlife communities at the landscape scale. 
Ecological Informatics 90, 103294. 
doi.org/10.1016/j.ecoinf.2025.103294

These models convert thier input into an embedding from a large audio model and 
do processing on top of that embedding
"""

from .model import ModelInput, ModelOutput
from torch import nn, Tensor
from perch_hoplite.zoo import model_configs
from .model import Model, ModelInput, ModelOutput, has_required_inputs
from transformers import PretrainedConfig

## Common Classes

class EmbeddingModel():
    def embed(self):
        raise NotImplementedError()

class EmbeddingInput(ModelInput):
    model = EmbeddingModel()
    embedding_size = 0

    def __init__(self, 
        labels,
        waveform = None,
        spectrogram = None):
        super().__init__(labels, waveform, spectrogram)

        self["embedding"] = self.model.embed(waveform)

## Unique Models

class PerchEmbeddings(EmbeddingModel):
    model = model_configs.load_model_by_name('perch_8')
    def embed(self, waveforms):
        # embeddings = [
        #     self.model.embed(waveform).embeddings[0]
        #     for waveform in waveforms
        # ]
        return waveforms

class PerchEmbeddingInput(EmbeddingInput):
    model = PerchEmbeddings()
    embedding_size = 1280


class FewShotModelConfig(PretrainedConfig):
    """Config for Timm Model Zoo Models!"""
    def __init__(
        self,
        num_classes=200,
        **kwargs
    ):
        """Creates Config.

        Args:
           
        """
        self.num_classes = num_classes
        super().__init__(**kwargs)

class PerchFewShotModel(Model, nn.Module):
    def __init__(
        self,
        config: FewShotModelConfig
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

        self.input_format = PerchEmbeddingInput
        self.output_format = ModelOutput

        self.config = config
        assert config.num_classes > 0

        # TODO BUILD MLP
        self.linear = nn.Linear(self.input_format.embedding_size, config.num_classes)

        # TODO USE CUSTOM LOSS FOR FEW SHOW LEARNING
        self.loss = nn.BCEWithLogitsLoss()

    @has_required_inputs()
    def forward(self, x: PerchEmbeddingInput):
        # Use perch to create embeddings
        embeddings = Tensor(x.model.model.embed(x["waveform"].cpu()).embeddings).to(x["waveform"].device)
          
        logits = self.linear(embeddings).squeeze(1)
        loss = self.loss(logits, x["labels"])

        return ModelOutput(
            logits=logits,
            embeddings=embeddings,
            loss=loss,
            labels=x["labels"]
        )
        

