"""Trains a Mutliclass Model with Pytorch and Huggingface

This script can be used to run experiments with different
models and datasets to create any model for bioacoustic classification

It is intended this script to be heavily modified with each experiment
(say one wants to use a different dataset, one should copy this and change the extractor!)

Usage:
    $ python train.py /path/to/config.yml

config.yml should contain frequently changed hyperparameters
"""

import argparse
import yaml

from pyha_analyzer import PyhaTrainer, PyhaTrainingArguments

from whoot_model_training.data_extractor import buowset_extractor
from whoot_model_training.models import TimmModel, TimmInputs
from whoot_model_training.preprocessors import SpectrogramModelInputPreprocessors

## TODO ALLOW USER TO SELECT THIS
## TODO MAKE DISTRIBUTED TRAINING POSSIBLE
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def parse_config(config_path: str) -> dict:
    """wrapper to parse config

    Args: 
        config_path (str): path to config file for training!
    
    returns: 
        (dict): hyperparameters parameters 
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def train(config_path):
    """Highest level logic for training

    Does the following:
    - Formats the dataset into an AudioDataset
    - Prepares preprocessing for each audio clip
    - Builds the model
    - Configures and runs the trainer
    - Runs evaluation 

    Args: 
        config_path (str): path to config file for training!
    """
    
    config = parse_config(config_path)

    # Extract the dataset
    ds = buowset_extractor(
        metadata_csv=config["metadata_csv"],
        parent_path=config["data_path"],
        output_path=config["hf_cache_path"],
    )

    # Create the model
    model = TimmModel(num_classes=ds.get_num_classes())

    # Preprocessors (No augmentation)!
    # We define here what the model reads
    preprocessor = SpectrogramModelInputPreprocessors(
        TimmInputs, duration=3, class_list=ds.get_class_labels()
    )

    ds["train"].set_transform(preprocessor)
    ds["valid"].set_transform(preprocessor)
    ds["test"].set_transform(preprocessor)

    # Run training
    args = PyhaTrainingArguments(working_dir="working_dir")
    
    # REQUIRED ARGS (DO NOT CHANGE VALUES TODO ADD TO TRAINER DIRECTLY)
    args.label_names = ["labels"]
    args.remove_unused_columns = False

    # OPTIONAL ARGS
    args.num_train_epochs = 2
    args.eval_steps = 10
    args.dataloader_num_workers = 36
    args.per_device_train_batch_size = 16
    args.per_device_eval_batch_size = 16
    args.run_name = "testing"
    args.report_to = "comet_ml"  # Blocks wandb


    print(args.accelerator_config.even_batches)
   

    trainer = PyhaTrainer(
        model=model,
        dataset=ds,
        training_args=args,
        logger=None,
    )

    print(trainer.evaluate(eval_dataset=ds["valid"], metric_key_prefix="TEST FOR METRICS"))
    # trainer.train()
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Input config path")
    parser.add_argument("config", type=str, help="Path to config.yml")
    args = parser.parse_args()
    train(args.config)
