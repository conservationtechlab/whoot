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

from whoot_model_training.trainer import WhootTrainer, WhootTrainingArguments
from whoot_model_training.data_extractor import buowset_extractor
from whoot_model_training.models import TimmModel, TimmInputs
from whoot_model_training import CometMLLoggerSupplement

from whoot_model_training.preprocessors import SpectrogramModelInputPreprocessors
from pyha_analyzer.preprocessors import MixItUp, ComposeAudioLabel
from audiomentations import Compose, AddColorNoise, AddBackgroundNoise, PolarityInversion, Gain

import comet_ml

## TODO ALLOW USER TO SELECT THIS
## TODO MAKE DISTRIBUTED TRAINING POSSIBLE
import os 


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


def train(config):
    """Highest level logic for training

    Does the following:
    - Formats the dataset into an AudioDataset
    - Prepares preprocessing for each audio clip
    - Builds the model
    - Configures and runs the trainer
    - Runs evaluation 

    Args: 
        config (dict): the config used for training. Defined in yaml file
    """

    # Extract the dataset
    ds = buowset_extractor(
        metadata_csv=config["metadata_csv"],
        parent_path=config["data_path"],
        output_path=config["hf_cache_path"],
    )

    # Create the model
    run_name =  "efficientnet_b1_augmented_mixitup_gain"
    model = TimmModel(timm_model="efficientnet_b1", num_classes=ds.get_num_classes())

    # Preprocessors

    # Augmentations
    # TODO: Design better system for saving and reproducing augmentation parameters
    wav_augs = ComposeAudioLabel([
        # AddBackgroundNoise( #We don't have background noise yet...
        #     sounds_path="data_birdset/background_noise",
        #     min_snr_db=10,
        #     max_snr_db=30,
        #     noise_transform=PolarityInversion(),
        #     p=0.8
        # ),
        Gain(
            min_gain_db = -12,
            max_gain_db = 12,
            p = 0.8
        ),
        MixItUp(
            dataset_ref=ds["train"],
            min_snr_db=10,
            max_snr_db=30,
            noise_transform=PolarityInversion(),
            p=0.8
        )
    ])

    # We define here what the model reads
    train_preprocessor = SpectrogramModelInputPreprocessors(
        TimmInputs, duration=3, class_list=ds.get_class_labels(), augment=wav_augs
    )

    preprocessor = SpectrogramModelInputPreprocessors(
        TimmInputs, duration=3, class_list=ds.get_class_labels()
    )

    ds["train"].set_transform(train_preprocessor)
    ds["valid"].set_transform(preprocessor)
    ds["test"].set_transform(preprocessor)

    # Run training
    args = WhootTrainingArguments(run_name=run_name)
    
    # REQUIRED ARGS (DO NOT CHANGE VALUES TODO ADD TO TRAINER DIRECTLY)
    args.label_names = ["labels"]
    args.remove_unused_columns = False

    # OPTIONAL ARGS
    args.num_train_epochs = 2
    args.eval_steps = 20
    args.per_device_train_batch_size = 32
    args.per_device_eval_batch_size = 32
    args.dataloader_num_workers = 36
    args.run_name = run_name
    args.report_to = "comet_ml"  # Blocks wandb


    print(args.accelerator_config.even_batches)
   

    trainer = WhootTrainer(
        model=model,
        dataset=ds,
        training_args=args,
        logger=CometMLLoggerSupplement(
            augmentations = wav_augs,
            name = args.run_name
        ),
        ignore_keys=["predictions", "labels", "embeddings", "loss"]
    )

    trainer.train()
    # print(trainer.evaluate(eval_dataset=ds["valid"], metric_key_prefix="TEST FOR METRICS"))
    

def init_env(config: dict):
    print(config)
    os.environ["COMET_PROJECT_NAME"] = config["COMET_PROJECT_NAME"]
    os.environ["CUDA_VISIBLE_DEVICES"] = config["CUDA_VISIBLE_DEVICES"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Input config path")
    parser.add_argument("config", type=str, help="Path to config.yml")
    args = parser.parse_args()
    config = parse_config(args.config)

    init_env(config)
    train(config)
