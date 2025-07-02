from datetime import datetime
import os

from pyha_analyzer import PyhaTrainingArguments
from pyha_analyzer import PyhaTrainer

from .metrics import WhootMutliClassMetrics
from .dataset import AudioDataset


class WhootTrainingArguments(PyhaTrainingArguments):
    def __init__(self, run_name):
        DEFAULT_MODEL_CHECKPOINTS = "model_checkpoints"
        checkpoint_created_at = datetime.now().strftime("%m_%d_%Y_%H:%M:%S")
        super().__init__(os.path.join(f"{DEFAULT_MODEL_CHECKPOINTS}",
                                      f"{run_name}_{checkpoint_created_at}"))


class WhootTrainer(PyhaTrainer):
    def __init__(
        self,
        model,
        dataset: AudioDataset,
        training_args=None,
        logger=None,
        data_collator=None,
        preprocessor=None,
        ignore_keys=...
    ):

        metrics = WhootMutliClassMetrics(dataset.get_class_labels().names)

        print("LOGGING NEW METRICS... HOPEFULLY")

        super().__init__(
            model,
            dataset,
            metrics,
            training_args,
            logger,
            data_collator,
            preprocessor,
            ignore_keys
        )
