"""Everything needed to train!

given a model and a dataset

WhootTrainingArguments: A container for the
    many many args for WhootTrainer

WhootTrainer: The class that is going to run training
"""

from datetime import datetime
import os

from pyha_analyzer import PyhaTrainingArguments
from pyha_analyzer import PyhaTrainer

from .metrics import WhootMutliClassMetrics
from .dataset import AudioDataset
from .models import Model
import torch
from tqdm import tqdm


class WhootTrainingArguments(PyhaTrainingArguments):
    """Holds arguments use for training."""
    def __init__(self,
                 run_name: str,
                 subproject_name: str = "TESTING",
                 dataset_name: str = "DS_404"):
        """Create Arguments.

        Args:
            run_name (str): name of the current run
            subproject_name (str): name of subproject
                These experiments are a part of
            dataset_name (str): name of dataset
                used for model experiments
        """
        assert subproject_name is not None
        assert dataset_name is not None

        default_checkpoint_path = "model_checkpoints"
        checkpoint_created_at = datetime.now().strftime("%m_%d_%Y_%H:%M:%S")

        # run_name is name of the model
        # task_name is name of the model task and dataset trained
        self.run_name = f"{subproject_name}_{dataset_name}_{run_name}"
        self.task_name = f"{subproject_name}_{dataset_name}"

        print(
            f"Starting training on {dataset_name} for {subproject_name}"
        )

        super().__init__(os.path.join(f"{default_checkpoint_path}",
                                      f"{run_name}_{checkpoint_created_at}"))

        # Required for whoot: override defaults in PyhaTrainingArguments
        self.label_names = ["labels"]
        self.remove_unused_columns = False
        self.report_to = "comet_ml"


class WhootTrainer(PyhaTrainer):
    """Trainers run the training of a model."""
    # WhootTrainer is ment to mimic the huggingface trainer
    # Including number of arguments
    # Aside, we really should consider how useful R0913,R0917 is...

    # pylint: disable-next=R0913,R0917
    def __init__(
        self,
        model: Model,
        dataset: AudioDataset,
        training_args: WhootTrainingArguments = None,
        logger=None,
        preprocessor=None,
    ):
        """Creates a trainer to hold training setup.

        Args:
            model (Model): a pytorch model for training
                should inherit from BaseModel see `models/model.py`
            dataset (AudioDataset): A canonical audio dataset
                Ideally attached some a preprocessor and returns ModelInputs
            training_args (WhootTrainingArugments):
                All the parameters that define training
            logger (CometMLLoggerSupplement):
                Class that adds additional logging
                On top of logging done by PyhaTrainer
            preprocessor (PreProcessorBase):
                Preprocessor used for formatting the data
        """
        metrics = WhootMutliClassMetrics(dataset.get_class_labels().names)

        if logger is not None:
            logger.log_task(training_args.task_name)

        super().__init__(
            model,
            dataset,
            metrics,
            training_args,
            logger,
            None,  # Data Collator, about to be deprecated
            preprocessor,
            model.output_format.ignore_keys
        )

    def predict(
            self,
            test_dataset: AudioDataset,
            ignore_keys=None,
            metric_key_prefix: str = "test"
    ):
        """Run Inferance on a given dataset.

        Allows for getting predicted outputs to label a new dataset
        Args:
            test_dataset (AudioDataset): dataset to get preds from
            This has labels but they are meaningless in this method
            ignore_keys: N/A
            metric_key_prefix: str = "test"
        Returns: test_dataset with a new col: "pred"
        """
        test_dataloader = self.get_test_dataloader(test_dataset)

        preds = []
        for batch in tqdm(test_dataloader):
            preds.append(self.model(
                self.model.input_format(**batch)
            )["logits"].detach().cpu())

        dataset = test_dataset.to_dict()
        dataset["pred"] = torch.concat(preds).detach().numpy()
        return dataset
