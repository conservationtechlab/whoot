from abc import ABC, abstractmethod

from torch import nn
import numpy as np

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
            loss: np.array | None = None
        ):
        self.embeddings = embeddings
        self.logits = logits
        self.loss = loss


class ModelInput(ABC):
    """ModelInput

    Spefifies Input Types
    Hopefully should help standardize formatting for models

    Inspired by HuggingFace Models and Tokenizers

    Developer: Reccommend for each Model, to have an assocaited ModelInput class
    """

    def __init__(
        self, 
        waveform: np.array | None = None, 
        spectrogram: np.array | None = None,
    ):
        self.waveform = waveform
        self.spectrogram = spectrogram


class Model(ABC):
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

    In PyTorch models, this is the same forward function
    We just apply the convention for non Pytorch models,

    TODO: Some things to concern
    - 
    Args:
        x: Any 

    Returns:
        ModelOutput: dict, a dictionary like object that describes 
    """
    @abstractmethod
    def forward(self, x: ModelInput) -> ModelOutput:
        pass


    """
    Notes on design for the future

    - Should model implement a way to save/load model to/form disk
    
    """