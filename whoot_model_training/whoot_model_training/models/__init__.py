"""a Bioacoustic Model Zoo!

Example:
    `from whoot_model_training.models import TimmModel
"""

from .timm_model import TimmModel, TimmInputs, TimmModelConfig
from .model import Model, ModelInput, ModelOutput

__all__ = [
    "TimmModel",
    "TimmInputs",
    "TimmModelConfig",
    "Model",
    "ModelInput",
    "ModelOutput"
]
