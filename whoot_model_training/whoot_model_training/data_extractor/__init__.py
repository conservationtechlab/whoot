"""A zoo for extractors

Extractors convert raw data into AudioDatasets
Ideally you make a new Extractor for each new raw dataset
"""
from .buowset_extractor import (
    buowset_extractor,
    buowset_binary_extractor,
)
from .esc50_extractor import esc50_extractor

__all__ = ["buowset_extractor", "buowset_binary_extractor", "esc50_extractor"]
