from torch import nn
from abc import ABC, abstractmethod
import numpy as np
import typing



class Model(ABC):
    """
    Gets an embedding for the model

    This can be the final layer of a model backbone
    or a set of useful features

    Returns
        embedding
    """
    @abstractmethod
    def get_embeddings(x) -> np.array:



