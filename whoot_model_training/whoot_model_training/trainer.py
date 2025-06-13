# """
#     The Trainer holds the main training loop, validation loop, and can run evaluation

#     There are some off the shelf options, such as the hugging face Trainer
#     Which is in use by https://github.com/UCSD-E4E/pyha-analyzer-2.0/

#     However, It can be difficult to fit input to perfectly
#     match what the hugging face trainer expects
#     And we are unlikely to use all the bells and whistles offered by hugging face.

#     So this SimpleTrainer can get the job spefifically for whoot done
#     With fewer bells and whistles
#     This should hopefully make debugging easier in the future and
#     keep the repo focused on whoot applications
# """

# import torch
# from transformers import TrainingArguments

# from .models.model import Model, ModelOutput
# from .dataset import AudioDataset
# from pyha_analyzer.metrics.classification_metrics import AudioClassificationMetrics


# class WhootTrainingArguments(TrainingArguments):
#     def __init__(self, working_dir):
#         super().__init__(working_dir)
#         self.logging_steps = 10
#         self.eval_steps = 100
#         self.per_device_train_batch_size = 64
#         self.per_device_eval_batch_size = 64
#         self.dataloader_num_workers = 4
#         self.eval_accumulation_steps = 10

# class WhootTrainer():
#     def __init__(
#             self,
#             model: Model,
#             dataset: AudioDataset,
#             metrics: AudioClassificationMetrics = None,
#             training_args: WhootTrainingArguments = None,
#             data_collator=None,
#             preprocessor=None,
#         ):

#         self.model = model
#         self.dataset = dataset
#         self.dataloaders = self._get_dataloaders(dataset)
#         self.metrics = metrics

#     def run_metrics(self, output_batches:list[ModelOutput]):
#         out = ModelOutput.concat(output_batches)
#         metrics = self.metrics(out.to_hugging_face())
#         print(metrics)

#     def run_step(self, batch, training=True):


#     def run_loop(self, split):
#         for i in range():

#     def train(self):

#     def evaluate(self):


