from datetime import datetime
import os

from pyha_analyzer import PyhaTrainingArguments
from pyha_analyzer import PyhaTrainer as WhootTrainer
# In case we want to extend the current Trainer, lets temporarily create WhootTrainer!

class WhootTrainingArguments(PyhaTrainingArguments):
    def __init__(self, run_name):
        DEFAULT_MODEL_CHECKPOINTS = "model_checkpoints"
        checkpoint_created_at = datetime.now().strftime("%m_%d_%Y_%H:%M:%S")
        super().__init__(os.path.join(f"{DEFAULT_MODEL_CHECKPOINTS}", 
                                      f"{run_name}_{checkpoint_created_at}"))
        