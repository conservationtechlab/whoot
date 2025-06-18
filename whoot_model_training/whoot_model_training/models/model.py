from abc import ABC, abstractmethod
from functools import wraps
from collections import UserDict

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
        def wrapper(self, *args, **kwarg):
            #assert isinstance(x, self.input_format) #TODO FIX
            model_output = forward(self, *args, **kwarg)
            #assert isinstance(model_output, self.output_format)

            return model_output

        return wrapper

    return decorator

# TODO: Simplify, most of this should have been done by UserDict...
class ModelOutput(dict, UserDict):
    """ModelOutput

    Object that stores the output of a model
    This allows for standardizing model outputs
    So upstream applications don't need to change for spefific models

    Inspired by HuggingFace Models

    Developer: Reccommend for each Model, to have an assocaited ModelOutput class
    """

    def __init__(
        self,
        _map: dict | None = None,
        logits: np.ndarray | None = None,
        embeddings: np.ndarray | None = None,
        labels: np.ndarray | None = None,
        loss: np.ndarray | None = None,
    ):  
        super(UserDict).__init__()
        self._main_keys = ["logits", "embeddings", "labels", "loss"]

        self.logits = logits
        self.embeddings = embeddings
        self.labels = labels
        self.loss = loss
        self.data = {
            "logits": self.logits, 
            "embeddings": self.embeddings, 
            "labels": self.labels,
            "loss": self.loss
        }
        if _map is not None:
            for key, value in _map:
                self[key] = value

        assert isinstance(self, dict)

    def to_hugging_face(self):
        return {
            "predictions": self.logits,
            "label_ids": [self.labels],
        }

    @classmethod
    def concat(list_of_outputs: list):
        return ModelOutput(
            logits=torch.vstack([out.logits for out in list_of_outputs]),
            embeddings=torch.vstack([out.embeddings for out in list_of_outputs]),
            loss=torch.vstack([out.loss for out in list_of_outputs]),
            labels=torch.vstack([out.labels for out in list_of_outputs]),
        )
    
    def __len__(self) -> int:
        """
            Count the number of batches in this system

            returns batch_size int
        """
        return len(self.labels)
    
    def __setitem__(self, key, value):
        if key in self._main_keys:
            self.__setattr__(key, value)
            self.data[key] = value
    
    def __getitem__(self, key):
        return self.__getattribute__(key)
    
    def __repr__(self):
        return str(self.data)
    
    def items(self):
        data = self.data.items()
        return ((col, value) for col, value in data if value is not None )
    
    def keys(self):
        return [key for key, _ in self.items()]
    
    def __iter__(self):
        return iter(self.keys())
    
    def __contains__(self, key):
        return key in self.data
    
    


class ModelInput(ABC, UserDict):
    """ModelInput

    Spefifies Input Types
    Hopefully should help standardize formatting for models

    Inspired by HuggingFace Models and Tokenizers

    Developer: Reccommend for each Model, to have an assocaited ModelInput class
    """

    def __init__(
        self,
        labels: np.ndarray,
        waveform: np.ndarray|None = None,
        spectrogram: np.ndarray |None = None,
    ):
        self.waveform = waveform
        self.spectrogram = spectrogram
        self.labels = labels
        self.data = {
            "labels": self.labels, 
            "waveform": self.waveform, 
            "spectrogram": self.spectrogram
        }
        self._main_keys = ["labels", "spectrogram", "waveform"]


    def to_tensor(self, device="cpu"):
        self.waveform = Tensor(self.waveform, device=device)
        self.spectrogram = Tensor(self.spectrogram, device=device)
        self.labels = Tensor(self.labels, device=device)

    def __len__(self) -> int:
        """
            Count the number of batches in this system

            returns batch_size int
        """
        return len(self.labels)
    
    def __setitem__(self, key, value):
        if key in self._main_keys:
            self.__setattr__(key, value)
            self.data[key] = value
    
    # TODO: There might be a smarter way to do something like this...
    def __getitem__(self, key):
        return self.__getattribute__(key)
    
    def __repr__(self):
        return str(self.data)
    
    def items(self):
        data = super().items()
        return ((col, value) for col, value in data if value is not None )

"""
BaseModel Class for Whoot
"""
class Model(BaseModel):
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
    @has_required_inputs()
    def forward(self, x: ModelInput) -> ModelOutput:
        pass

    """
    Notes on design for the future

    - Should model implement a way to save/load model to/form disk
    
    """
