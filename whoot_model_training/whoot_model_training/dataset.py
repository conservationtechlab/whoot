"""
Pulled from:
https://github.com/UCSD-E4E/pyha-analyzer-2.0/blob/main/pyha_analyzer/dataset.py
Key idea is we define a generic AudioDataset with uniform features

Using an Arrow Dataset from Hugging Face's dataset library because
- Cool audio features https://huggingface.co/docs/datasets/en/audio_process
- Faster than pandas, better at managing memory
"""

from datasets import DatasetDict, ClassLabel

DEFAULT_COLUMNS = ["labels", "audio"]


class AudioDataset(DatasetDict):
    """
    AudioDataset Class

    If your dataset is an AudioDataset, it can be read by
    the rest of the system

    Behind the scenes, this is a Apache Arrow Dataset Dict
    (via hf library) where
    each key is a split of the data (test/train/valid)
    and the value is an arrow dataset
    with at a minimum 2 columns:
    - labels (Sequence of class labels, such as [0,10])
    - audio (Audio Column type from hugging face)
    """
    def __init__(self, ds: DatasetDict):
        self.validate_format(ds)
        super().__init__(ds)

    def validate_format(self, ds: DatasetDict):
        """Validates dataset is correctly formatted and ready to be used for
        training

        Raises:
            AssertionError if dataset is not correctly formatted.
        """
        for split in ds.keys():
            dataset = ds[split]
            for column in DEFAULT_COLUMNS:
                state = (
                    f"The column `{column}` is missing from dataset split `{
                        split}`. Required by system"
                )
                assert column in dataset.features, state

    def get_num_classes(self):
        """
            Returns:
                (int): the number of classes in this dataset
        """
        return self["train"].features["labels"].feature.num_classes

    def get_number_species(self) -> int:
        """
            PyhaAnalyzer uses `get_number_species` for getting class count
            This... isn't always the case that the dataset is species only
            (could have calls!)
            To support legacy PyhaAnalyzer, we therefore have this function.

            This should be deprecated in future versions of PyhaAnalyzer

            return
                (int): number of classes
        """
        return self.get_num_classes()

    def get_class_labels(self) -> ClassLabel:
        """Class mapping for this dataset

        A common problem is when moving between datasets
        creating mappings between classes
        This aims to help standardize that by being
        able to get the classLabels for this dataset

        Returns:
            (ClassLabel): Mapping of all the names of
                the labels to their index.
        """
        return ClassLabel(names=self["train"].features["labels"].feature.names)
