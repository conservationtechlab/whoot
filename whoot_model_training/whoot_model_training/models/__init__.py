"""a Bioacoustic Model Zoo!

Example:
    `from whoot_model_training.models import TimmModel
"""

from .timm_model import TimmModel, TimmInputs, TimmModelConfig
from .model import Model, ModelInput, ModelOutput
from .few_shot_model import PerchEmbeddingInput, PerchFewShotModel, FewShotModelConfig

__all__ = [
    "TimmModel",
    "TimmInputs",
    "TimmModelConfig",
    "Model",
    "ModelInput",
    "ModelOutput",
    "PerchEmbeddingInput",
    "PerchFewShotModel",
    "FewShotModelConfig"
]
