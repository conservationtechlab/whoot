import os
import argparse
import yaml

from whoot_model_training.trainer import WhootTrainer, WhootTrainingArguments
from whoot_model_training.data_extractor import buowset_extractor
from whoot_model_training.models import TimmModel, TimmInputs, TimmModelConfig
from whoot_model_training import CometMLLoggerSupplement

from whoot_model_training.preprocessors import (
    MelModelInputPreprocessor
)

# New Dataset
"""Trains a Mutliclass Model with Pytorch and Huggingface.

This script can be used to run experiments with different
models and datasets to create any model for bioacoustic classification

It is intended this script to be heavily modified with each experiment
(say one wants to use a different dataset, one should copy this and change the
extractor!)

Usage:
    $ python train.py /path/to/config.yml

config.yml should contain frequently changed hyperparameters
"""
import os
import argparse
import yaml

from whoot_model_training.trainer import WhootTrainer, WhootTrainingArguments
from whoot_model_training.data_extractor import buowset_extractor
from whoot_model_training.models import TimmModel, TimmInputs, TimmModelConfig
from whoot_model_training import CometMLLoggerSupplement
from train import parse_config, init_env

from whoot_model_training.preprocessors import (
    MelModelInputPreprocessor
)

def test(config, model_name=""):
    """Highest level logic for inferance.

    Does the following:
    - Formats the dataset into an AudioDataset
    - Prepares preprocessing for each audio clip
    - Builds the model
    - Configures and runs the trainer
    - Runs evaluation

    Args:
        config (dict): the config used for training. Defined in yaml file
        TODO
    """
    # Extract a new dataset
    ds = buowset_extractor(
        metadata_csv=config["metadata_csv"],
        parent_path=config["data_path"],
        output_path=config["hf_cache_path"],
    )

    # Create the model
    model = TimmModel.from_pretrained(model_name)

    preprocessor = MelModelInputPreprocessor(
        TimmInputs, duration=3
    )

    ds["train"].set_transform(preprocessor)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Input config path")
    parser.add_argument("config", type=str, help="Path to config.yml")
    parser.add_argument(
        "model_name", 
        help="path to weights or hugging face repo id",
        default="/home/sean/whoot/model_checkpoints/buowset1.1_efficientnet_b1_08_15_2025_13:16:04/checkpoint-1580")
    args = parser.parse_args()
    _config = parse_config(args.config)

    init_env(_config)
    test(_config, model_name=args.model_name)
