"""Abstract Model Class for training.

Any model trained with this repo SHOULD inherit from these classes found here

There are 3 main classes
- ModelInput: dict-like class that define required input params to function
- ModelOutput: dict-like class that defines the output from the model
- Model: A PyTorch nn.Module class

See timm_model.py for example about how these classes can be implemented.
"""

from abc import abstractmethod
from functools import wraps
from collections import UserDict

from pyha_analyzer.models.base_model import BaseModel
from transformers import PreTrainedModel, PretrainedConfig
import numpy as np


def has_required_inputs():
    """Wrapper for formatting input for a given Model!

    Checks to make sure a model is passed in the correct input
    format, and returns the correct output format.

    Usually this is defined by `model.input_format` and
    `model.output_format`

    MUST ALWAYS WRAP FORWARD FUNCTION OF MODEL
    """

    def decorator(forward):
        @wraps(forward)
        def wrapper(self, x=None, **kwarg):
            # During training, data is passed in as kwargs, (**ModelInput)
            # due to how hugging face is designed
            # this can be confusing if you are making custom models
            # During inference, data is passed in as x, (ModelInput)
            if x is None:
                # ... but during training we just have the model
                # pretend like it was passed in a ModelInput
                x = self.input_format.from_dict(kwarg)

            assert isinstance(x, self.input_format)
            model_output = forward(self, x)
            assert isinstance(model_output, self.output_format)

            return model_output

        return wrapper

    return decorator


class ModelOutput(dict, UserDict):
    """ModelOutput.

    Object that stores the output of a model
    This allows for standardizing model outputs
    So upstream applications don't need to change for specific models

    Inspired by HuggingFace Models

    Developer: recommended for each Model, to have an associated
        ModelOutput class
    """

    # ignore some of the outputs when computing metrics
    # When overwriting DON"T FORGET TO INCLUDE THIS
    ignore_keys = ["predictions", "labels", "embeddings", "loss"]

    def __init__(
        self,
        _map: dict | None = None,
        logits: np.ndarray | None = None,
        embeddings: np.ndarray | None = None,
        labels: np.ndarray | None = None,
        loss: np.ndarray | None = None,
    ):
        """Create a new output to a model!

        Args:
            logits: raw output from model
            embeddings: some latent space encoding of data
                Useful for transfer learning!
            labels: labels for computing metrics
            loss: loss as computed by the model
        """
        super().__init__(
            {
                "predictions": logits,
                "logits": logits,
                "labels": [labels],
                # "label_ids": [labels],
                "embeddings": embeddings,
                "loss": loss,
            }
        )

    def items(self):
        """Get all items in dict.

        But only if they are defined (not null)!
        """
        return [(key, value) for (key, value) in super().items() if value is not None]


class ModelInput(UserDict, dict):
    """ModelInput.

    Specifies Input Types
    Hopefully should help standardize formatting for models

    Inspired by HuggingFace Models and Tokenizers

    Developer: recommended for each Model, to have an
    associated ModelInput class
    ALWAYS HAS A LABEL CATEGORY
    """

    def __init__(
        self,
        labels: np.ndarray,
        waveform: np.ndarray | None = None,
        spectrogram: np.ndarray | None = None,
    ):
        """Create a new input to a model!

        Args:
            labels: one_hot encoded labels
            waveform: raw audio signal
            spectrogram: 2d matrix to represent the waveform
        """
        super().__init__(
            {"labels": labels, "waveform": waveform, "spectrogram": spectrogram}
        )

    def items(self):
        """Get all items in dict.

        But only if they are defined (not null)!
        """
        return [(key, value) for (key, value) in super().items() if value is not None]

    @classmethod
    def from_dict(cls, some_input: dict):
        """Recreates input for models!

        Sometimes inputs are given as kwargs
        So lets recreate correct inputs for model
        via building from a dictionary!
        """
        spectrogram, waveform = None, None
        labels = some_input["labels"]
        if "spectrogram" in some_input:
            spectrogram = some_input["spectrogram"]
        if "waveform" in some_input:
            waveform = some_input["waveform"]

        assert spectrogram is not None or waveform is not None

        return cls(labels, spectrogram=spectrogram, waveform=waveform)


class Model(PreTrainedModel, BaseModel):
    """BaseModel Class for Whoot."""

    def __init__(self):
        """Creates a basic model format.

        Anytime you create a new model, check if you need
        to specify an input and output format for this model!
        """
        self.input_format = ModelInput
        self.output_format = ModelOutput
        PreTrainedModel.__init__(self, PretrainedConfig())

        assert hasattr(self.forward, "__wrapped__"), (
            "Please put `@has_required_inputs()",
            "on the forward function of the model",
        )

    def get_embeddings(self, x: ModelInput) -> np.array:
        """Gets an embedding for the model.

        This can be the final layer of a model backbone
        or a set of useful features

        Args
            x: Any | Either np.array or Torch.Tensor, the input for the model

        Returns
            embedding: np.array,
                some embedding vector representing the input data
        """
        return self.forward(**x).embeddings

    @abstractmethod
    @has_required_inputs()
    def forward(self, x: ModelInput) -> ModelOutput:
        """Runs some input x through the model.

        In PyTorch models, this is the same forward function logits
        We just apply the convention for non Pytorch models,
        Args:
            x: Any

        Returns:
            ModelOutput: dict, a dictionary like object that describes
        """

    def get_position_embeddings(self):
        """Required by PretrainedModel, not needed for our work yet!"""
        print("this model doesn't support position_embeddings")

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        """Required by PretrainedModel, not needed for our work yet!"""
        print("this model doesn't support position_embeddings")
