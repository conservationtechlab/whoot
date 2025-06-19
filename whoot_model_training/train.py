

import argparse
import yaml

from pyha_analyzer import PyhaTrainer, PyhaTrainingArguments

from whoot_model_training.data_extractor import buowset_extractor
from whoot_model_training.models import TimmModel, TimmInputs
from whoot_model_training.preprocessors import SpectrogramModelInputPreprocessors

def parse_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def train(config_path):
    config = parse_config(config_path)

    # Extract the dataset
    ds = buowset_extractor(
        metadata_csv=config["metadata_csv"],
        parent_path=config["data_path"],
        output_path=config["hf_cache_path"]
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
    args.num_train_epochs = 2
    args.remove_unused_columns = False
    args.label_names = ["labels"]
    args.eval_steps = 20
    args.per_device_train_batch_size = 1
    args.per_device_eval_batch_size = 1
    args.dataloader_num_workers = 0
    args.run_name = "testing"
    args.report_to="none" #Blocks wandb

    trainer = PyhaTrainer(
        model=model,
        dataset=ds,
        training_args=args,
        logger=None,
        ignore_keys=["predictions", "labels", "embeddings", "loss"]
    )
    print(trainer.evaluate(eval_dataset=ds["test"], metric_key_prefix="Soundscape"))
   


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Input config path'
        )
    parser.add_argument('config', type=str,
                        help='Path to config.yml')
    args = parser.parse_args()
    train(args.config)