# pylint: skip-file
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

import argparse
import pickle

from whoot_model_training.trainer import WhootTrainer, WhootTrainingArguments
from whoot_model_training.data_extractor import raw_audio_extractor
from whoot_model_training.models import TimmModel, TimmInputs
from whoot_model_training import CometMLLoggerSupplement
from train import parse_config, init_env

from whoot_model_training.preprocessors import MelModelInputPreprocessor



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
        model_name (str): Model name for this run
    """
    # Extract a new dataset
    unlabel_audio_path = "/mnt/restorage/Audiomoth/Raw sound files/2024/RGCB/"
    ds = raw_audio_extractor(
        audio_parent_folder=unlabel_audio_path,
        output_folder="data/manual_buowset",
        chunk_duration=3,
    )

    # ds = buowset_extractor(
    #     metadata_csv=config["metadata_csv"],
    #     parent_path=config["data_path"],
    #     output_path=config["hf_cache_path"],
    # )

    # Create the model
    model = TimmModel.from_pretrained(model_name)

    preprocessor = MelModelInputPreprocessor(TimmInputs, duration=3)

    ds["train"].set_transform(preprocessor)
    # ds["valid"].set_transform(preprocessor)
    # ds["test"].set_transform(preprocessor)

    model_name = "efficientnet_b1"
    run_name = f"buowset1.1_{model_name}_ATTEMPT_TO_STUDY_NEW_DATA"

    # trainer = WhootTrainer._load_from_checkpoint(model_name)

    # Run training
    training_args = WhootTrainingArguments(
        run_name=run_name,
        subproject_name=config["SUBPROJECT_NAME"] + "_INFERANCE",
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
        logger=CometMLLoggerSupplement(
            augmentations=None,
            name=training_args.run_name
        ),
    )

    # print(ds["train"].shape, ds["test"].shape, ds["valid"].shape)
    # input()

    out = trainer.predict(ds["train"])
    print(out)
    with open(run_name + ".pkl", mode="wb") as f:
        pickle.dump(out, f)
    # trainer.evaluate(ds["test"], metric_key_prefix="test")
    # trainer.evaluate(ds["valid"], metric_key_prefix="valid")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Input config path")
    parser.add_argument("config", type=str, help="Path to config.yml")
    parser.add_argument(
        "--model_name",
        required=False,
        help="path to weights or hugging face repo id",
        default="/home/sean/whoot/checkpoint-4985",
    )
    args = parser.parse_args()
    _config = parse_config(args.config)

    init_env(_config)
    test(_config, model_name=args.model_name)
