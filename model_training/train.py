from pyha_analyzer import PyhaTrainer, PyhaTrainingArguments

from .data_extractor import buowset_extractor
from .models import TimmModel, TimmInputs
from .preprocessors import SpectrogramModelInputPreprocessors


# Extract the dataset
ds = buowset_extractor(
    metadata_csv="data.csv", parent_path="data_parent_path/data/", output=None
)

# Create the model
model = TimmModel(num_classes=ds.get_num_classes())

# Preprocessors (No augmentation)!
# We define here what the model reads
preprocessor = SpectrogramModelInputPreprocessors(
    TimmInputs, duration=5, class_list=ds["train"].features["labels"].feature.names
)
ds["train"].set_transform(preprocessor)
ds["valid"].set_transform(preprocessor)
ds["test"].set_transform(preprocessor)

# Run training
args = PyhaTrainingArguments(working_dir="working_dir")
args.num_train_epochs = 20
args.eval_steps = 20
args.run_name = "testing"

trainer = PyhaTrainer(
    model=model,
    dataset=ds,
    training_args=args,
    logger=None,
)
trainer.train()
trainer.evaluate(eval_dataset=ds["test"], metric_key_prefix="Soundscape")
