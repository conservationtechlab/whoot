# pylint: skip-file
"""Run a mutliclass model over a set of unlabeled data!

This scripts takes a folder of unlabeled data and the path
to a model checkpoint to get new data.

It is intended this script to be heavily modified with each diffrent model type
(say one wants to use a different model, one should copy this and change the
model type!)

Usage:
    $ python inference.py /path/to/config.yml
        --model_name /path/to/model/parent/dir/

config.yml should contain frequently changed hyperparameters
"""

import argparse
import datasets

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
    run_name = f"{model_name}_infer"
    # Will create predictions in model_checkpoint_folder

    ds = raw_audio_extractor(
        audio_parent_folder=unlabel_audio_path,
        output_folder="data/manual_buowset",
        chunk_duration=3,
        class_list=None
    )

    # Create the model
    model = TimmModel.from_pretrained(model_name)
    preprocessor = MelModelInputPreprocessor(TimmInputs, duration=3)
    ds["train"].set_transform(preprocessor)
    # This isn't technically train, but HF expects a train split

    # Run training
    training_args = WhootTrainingArguments(
        run_name=run_name,
        subproject_name=config["SUBPROJECT_NAME"] + "_INFERENCE",
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

    out = trainer.predict(ds["train"])
    # Pipeline requires a labels col
    # For inferance the "labels" are just an array of zeros
    # Therefore during inferance, "labels" are meaningless
    # Delete them to make it clearer to downstream users
    del out['labels']

    print(out)
    with open(run_name + ".pkl", mode="wb") as f:
         pickle.dump(out, f)
    # Below was tested with the pickle made from above
    ds = datasets.Dataset.from_dict(out)
    ds.save_to_disk(f"predictions/{run_name}")  # saves as a directory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Input config path")
    parser.add_argument("config", type=str, help="Path to config.yml")
    parser.add_argument(
        "--model_name",
        required=False,
        help="path to weights or hugging face repo id",
        default="Insert_Checkpoint_Here",
    )
    args = parser.parse_args()
    _config = parse_config(args.config)

    init_env(_config)
    test(_config, model_name=args.model_name)
