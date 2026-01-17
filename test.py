# # %%
# %load_ext autoreload 
# %autoreload 1

# %%

from  import WaveformInputPreprocessor
from whoot_model_training.whoot_model_training.models import HFInput, HFModel, HFModelConfig
from whoot_model_training.whoot_model_training.trainer import WhootTrainer, WhootTrainingArguments
from whoot_model_training.whoot_model_training.data_extractor import xc_extractor
from whoot_model_training.whoot_model_training import CometMLLoggerSupplement


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# %%
ds = xc_extractor(
    XC_dataset_json_path="xc_meta_aux.json",
    parent_path="/mnt/acoustics/san_diego_xc_aux/xeno-canto"
)



model = HFModel(HFModelConfig(num_classes=ds.get_number_species()))

# %%
# %%

input_wrapper = HFInput()

train_preprocessor = WaveformInputPreprocessor(
    input_wrapper, duration=3
)

preprocessor = WaveformInputPreprocessor(
    input_wrapper, duration=3
)

ds["train"].set_transform(train_preprocessor)
ds["valid"].set_transform(preprocessor)
ds["test"].set_transform(preprocessor)

print(ds.get_class_labels())

# run_name = "fewshot_test_birdmae"
# subproject_name = "fewshot_test"
# dataset_name = "san_diego_xc_aux_09_2025"

# training_args = WhootTrainingArguments(
#     run_name=run_name,
#     subproject_name=subproject_name,
#     dataset_name=dataset_name,
# )

# # COMMON OPTIONAL ARGS
# training_args.num_train_epochs = 100
# training_args.eval_steps = 2000
# training_args.per_device_train_batch_size = 16
# training_args.per_device_eval_batch_size = 16
# training_args.dataloader_num_workers = 16
# training_args.run_name = run_name
# training_args.learning_rate = 0.01
# training_args.save_strategy="steps", # Save at the end of each epoch
# training_args.save_total_limit=2 # Keep only the last 2 checkpoints

# trainer = WhootTrainer(
#     model=model,
#     dataset=ds,
#     training_args=training_args,
#     logger=CometMLLoggerSupplement(
#         augmentations=None,
#         name=training_args.run_name
#     ),
# )

# trainer.train()
# model.save_pretrained("model_checkpoints/fewshot_test_birdmae")

