"""
Pulled from https://github.com/UCSD-E4E/pyha-analyzer-2.0/blob/main/pyha_analyzer/dataset.py
Key idea is we define a generic AudioDataset with uniform features

Using an Arrow Dataset from Hugging Face's dataset library because
- Cool audio features https://huggingface.co/docs/datasets/en/audio_process
- Faster than pandas, better at manging memory

# TODO Use the default stuff from pyha-anaylzer
"""

from datasets import DatasetDict, ClassLabel
from torch.utils.data import DataLoader

DEFAULT_COLUMNS = ["labels", "audio"]


class AudioDataset(DatasetDict):
    def __init__(self, ds: DatasetDict):
        self.validate_format(ds)
        super().__init__(ds)

    def validate_format(self, ds: DatasetDict):
        for split in ds.keys():
            dataset = ds[split]
            for column in DEFAULT_COLUMNS:
                assert column in dataset.features, (
                    f"The column `{column}` is missing from dataset split `{split}`. Required by system"
                )

    def get_num_classes(
        self,
    ):  # NOTE: Assumes all labels are mutlilabel (the extra feature note)
        return self["train"].features["labels"].feature.num_classes
    
    """
        Legacy code had the method name `get_number_species`
    """
    def get_number_species(self):
        return self.get_num_classes()

    def get_class_labels(self):
        """
        Returns a new ClassLabel Object to make mapping easier between datasets
        """
        return ClassLabel(names=self["train"].features["labels"].feature.names)
