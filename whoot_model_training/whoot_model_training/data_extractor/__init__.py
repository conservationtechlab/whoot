"""A zoo for extractors.

Extractors convert raw data into AudioDatasets
Ideally you make a new Extractor for each new raw dataset
"""
from .buowset_extractor import (
    buowset_extractor,
    buowset_binary_extractor,
)
from .esc50_extractor import esc50_extractor
from .Jacuzzi_Olden_extractor import Jacuzzi_Olden_Extractor
from .xc_extractor import xc_extractor

__all__ = ["buowset_extractor", "buowset_binary_extractor", "esc50_extractor", "Jacuzzi_Olden_Extractor", "xc_extractor"]

def concat_dataset(datasetA, datasetB):
    for split in datasetA.keys():
        pass

        #TODO FIGURE OUT HOW TO SAFETLY COMBINE TWO DATASETS

        # labels
        # this is tricky, you need to check class names for union, then 
        # Apply annotations accordingly
        # maybe use a dict to handle classes in both datasets
        
        # Audio
        # should be able to merge
        
        # Metadata
        # Consider dropping all non-required columns, will make merge easier