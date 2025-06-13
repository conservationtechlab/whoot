from abc import ABC, abstractmethod
from functools import wraps

from pyha_analyzer.models.base_model import BaseModel
import torch
from torch import nn, Tensor
import numpy as np

"""
    Wrapper to check to make sure everything is setup properly
    Required before using PyhaTrainer
"""
def has_required_inputs():
    def decorator(forward):
        @wraps(forward)
        def wrapper(self, x):
            assert(isinstance(x, self.input_format))
            model_output = forward(self, x)
            assert(isinstance(model_output, self.output_format))

            return model_output
        return wrapper
    return decorator


class ModelOutput(ABC):
    """ModelOutput

    Object that stores the output of a model
    This allows for standardizing model outputs
    So upstream applications don't need to change for spefific models

    Inspired by HuggingFace Models

    Developer: Reccommend for each Model, to have an assocaited ModelOutput class
    """

    def __init__(
            self, 
            logits: np.array, 
            embeddings: np.array,
            labels: np.array | None = None,
            loss: np.array | None = None
        ):
        self.embeddings = embeddings
        self.logits = logits
        self.loss = loss
        self.labels = labels

    def to_hugging_face(self):
        return {
            "predictions": self.logits,
            "label_ids": [self.labels],
        }
    
    @classmethod
    def concat(list_of_outputs: list):
        return ModelOutput( 
            logits = torch.vstack([out.logits for out in list_of_outputs]),
            embeddings = torch.vstack([out.embeddings for out in list_of_outputs]),
            loss = torch.vstack([out.loss for out in list_of_outputs]),
            labels = torch.vstack([out.labels for out in list_of_outputs]),
        )
        


class ModelInput(ABC):
    """ModelInput

    Spefifies Input Types
    Hopefully should help standardize formatting for models

    Inspired by HuggingFace Models and Tokenizers

    Developer: Reccommend for each Model, to have an assocaited ModelInput class
    """

    def __init__(
        self, 
        labels: np.array,
        waveform: np.array | None = None, 
        spectrogram: np.array | None = None,
    ):
        self.waveform = waveform
        self.spectrogram = spectrogram
        self.labels = labels

    def to_tensor(self, device="cpu"):
        self.waveform = Tensor(self.waveform, device=device)
        self.spectrogram = Tensor(self.spectrogram, device=device)
        self.labels = Tensor(self.labels, device=device)

class Model(ABC, nn.Module, BaseModel):
    # TODO Define required class intance variables
    # Such as cirteron etc. 
    def __init__(self, *args, **kwargs):
        self.input_format = ModelInput
        self.output_format = ModelOutput
        super().__init__(*args, **kwargs)

    """
    Gets an embedding for the model

    This can be the final layer of a model backbone
    or a set of useful features

    Args
        x: Any | Either np.array or Torch.Tensor, is the input for the model

    Returns
        embedding: np.array, some embedding vector representing the input data
    """
    def get_embeddings(self, x: ModelInput) -> np.array:
        return self.forward(x).embeddings

    """
    Runs some input x through the model

    In PyTorch models, this is the same forward functionlogits
    We just apply the convention for non Pytorch models,

    TODO: Some things to concern
    - 
    Args:
        x: Any 

    Returns:
        ModelOutput: dict, a dictionary like object that describes 
    """
    @abstractmethod
    @has_required_inputs
    def forward(self, x: ModelInput) -> ModelOutput:
        pass


    """
    Notes on design for the future

    - Should model implement a way to save/load model to/form disk
    
    """