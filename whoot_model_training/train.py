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

from whoot_model_training.preprocessors import MelModelInputPreprocessor
from whoot_model_training.preprocessors.spectrogram_preprocessors import (
    SpectrogramParams,
)

# Uncomment for use with data augmentation
# from pyha_analyzer.preprocessors import MixItUp, ComposeAudioLabel
# from audiomentations import (
#   Compose, AddColorNoise,
#   AddBackgroundNoise, PolarityInversion, Gain
# )


def parse_config(config_path: str) -> dict:
    """Wrapper to parse config.

    Args:
        config_path (str): path to config file for training!

    Returns:
        (dict): hyperparameters parameters
    """
    config = {}
    with open(config_path, "r", encoding="UTF-8") as f:
        config = yaml.safe_load(f)
    return config


def train(config):
    """Highest level logic for training.

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
    model_name = "efficientnet_b1"

    run_name = f"buowset1.1_{model_name}"
    model_config = TimmModelConfig(
        timm_model=model_name, num_classes=ds.get_num_classes()
    )
    model = TimmModel(model_config)

    # Preprocessors

    # Uncomment if doing work with data augmentation
    # # Augmentations
    # wav_augs = ComposeAudioLabel([
    #     # AddBackgroundNoise( #We don't have background noise yet...
    #     #     sounds_path="data_birdset/background_noise",
    #     #     min_snr_db=10,
    #     #     max_snr_db=30,
    #     #     noise_transform=PolarityInversion(),
    #     #     p=0.8
    #     # ),
    #     Gain(
    #         min_gain_db = -12,
    #         max_gain_db = 12,
    #         p = 0.8
    #     ),
    #     MixItUp(
    #         dataset_ref=ds["train"],
    #         min_snr_db=10,
    #         max_snr_db=30,
    #         noise_transform=PolarityInversion(),
    #         p=0.8
    #     )
    # ])

    spectrogram_params = SpectrogramParams()
    # spectrogram_params = SpectrogramParams(
    #     n_mels = 224,
    #     hop_length = 286,
    # )
    # """Dataclass for spectrogram Parameters.

    # n_fft: (int) number of fft bins
    # hop_length (int) skip count
    # power: (float) usually 2
    # n_mels: (int) number of mel bins
    # """
    # n_fft: int = 2048
    # hop_length: int = 256
    # power: float = 2.0
    # n_mels: int = 256

    # Online preprocessors prepare data for training
    train_preprocessor = MelModelInputPreprocessor(
        TimmInputs, duration=3, spectrogram_params=spectrogram_params
    )

    preprocessor = MelModelInputPreprocessor(
        TimmInputs, duration=3, spectrogram_params=spectrogram_params
    )

    ds["train"].set_transform(train_preprocessor)
    ds["valid"].set_transform(preprocessor)
    ds["test"].set_transform(preprocessor)

    # Run training
    training_args = WhootTrainingArguments(
        run_name=run_name,
        subproject_name=config["SUBPROJECT_NAME"],
        dataset_name=config["DATASET_NAME"],
    )

    # COMMON OPTIONAL ARGS
    training_args.num_train_epochs = 5
    training_args.eval_steps = 100
    training_args.per_device_train_batch_size = 16
    training_args.per_device_eval_batch_size = 16
    training_args.dataloader_num_workers = 1
    training_args.run_name = run_name

    trainer = WhootTrainer(
        model=model,
        dataset=ds,
        training_args=training_args,
        logger=CometMLLoggerSupplement(augmentations=None, name=training_args.run_name),
    )

    trainer.train()
    model.save_pretrained("model_checkpoints/test")


def init_env(config: dict):
    """Sets up local environment for COMET-ML training logging.

    Args: config (dict): at a minimum this has the project name
        and CUDA devices that are allowed to be used.
    """
    print(config)
    os.environ["COMET_PROJECT_NAME"] = config["COMET_PROJECT_NAME"]
    os.environ["CUDA_VISIBLE_DEVICES"] = config["CUDA_VISIBLE_DEVICES"]
    check_for_comet = config["COMET_WORKSPACE"] is not None
    assert check_for_comet, "Make sure to add a COMET_WORKSPACE to config"
    os.environ["COMET_WORKSPACE"] = config["COMET_WORKSPACE"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Input config path")
    parser.add_argument("config", type=str, help="Path to config.yml")
    args = parser.parse_args()
    _config = parse_config(args.config)

    init_env(_config)
    train(_config)
