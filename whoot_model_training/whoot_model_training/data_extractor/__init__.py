"""A zoo for extractors

Extractors convert raw data into AudioDatasets
Ideally you make a new Extractor for each new raw dataset
"""
from .buowset_extractor import (
    buowset_extractor,
    buowset_binary_extractor,
)
from .panda_extractor import panda_extractor

__all__ = ["buowset_extractor", "buowset_binary_extractor", "panda_extractor"]
