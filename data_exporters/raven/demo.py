"""Demo for using Raven Exporter with a sample dataset.

Run from whoot_model_training after saving a cache of buowset
data. This will make selection tables out of the training data.
You may need to change the path in the example if you named
your dataset differently. Replace the parent folder with
the path to the original audio, this is where the Raven
selection tables will be stored.

There is also a commented out example of how you would load
a inference.py result .pkl after running inference on real
data.

Uses polars rather than Hugging Face Datasets because
for some of these groupby ops in here, it was more performant.
"""

import os
import datasets
import polars as pl
import tqdm


class RavenDataset():
    """RavenDataset.

    A data stucture to contain information pertaining to
    Raven Outputs.

    Args:
        ds (HF Dataset): Dataset of audio
            typically from training pipeline.
        duration (float): Duration in seconds of default label.
        sr (int): Sample rate of audio files if not given.
        label_col (string): Name of column with labels.
        dry_run (bool): If true, no files are made
            Filenames are printed to stdout.
    """
    def __init__(self,
                 dataset: datasets.Dataset,
                 duration: float = None,
                 sr: int = None,
                 label_col="labels",
                 dry_run=True):
        """Start building the Raven Dataset.

        Does not automatically make files without user
        asking for it.

        Allows for double checking for bad formatting.

        Args:
        ds (HF Dataset): Dataset of audio
            typically from training pipeline.
        duration (float): Duration in seconds of default label.
        sr (int): Sample rate of audio files if not given.
        label_col (string): Name of column with labels.
        dry_run (bool): If true, no files are made
            Filenames are printed to stdout.
        """
        self.ds = dataset
        self.raven_ds = None
        self.sr = sr
        self.label_col = label_col
        self.duration = duration
        self.build_dataset(dataset)
        self.header = [
            'Selection',
            'View',
            "Begin Time (s)",
            "End Time (s)",
            "Low Freq (Hz)",
            'High Freq (Hz)',
            'Delta Time (s)',
            'Delta Freq (Hz)',
            "Label"
        ]
        self.dry_run = dry_run

    def default_template_annotation_style(self, file_df: pl.DataFrame):
        """Format the annotations of a single file.

        Args:
            file_df (polars DataFrame): Annotations of a file
                Usually just converted from a HF dataset.

        Returns:
            pl.DataFrame: The Raven selection tables as a polars
                dataset.
        """
        sr = self.sr
        if "offset" in file_df.columns:
            offsets = file_df['offset']
            durations = file_df['duration']
        else:
            offsets = pl.lit(0)
            durations = pl.lit(self.duration)

        file_df = file_df.unnest("audio")

        file_df = file_df.with_row_index("Selection", offset=1)

        file_df = file_df.with_columns([
            pl.lit('Spectrogram 1').alias('View'),
            pl.col(self.label_col).alias('Label').cast(
                pl.List(pl.String)).list.join(","),
            offsets.alias('Begin Time (s)'),
            (offsets + durations).alias('End Time (s)'),
            pl.lit(0).alias('Low Freq (Hz)'),
            pl.lit(sr/2).alias('High Freq (Hz)'),
            durations.alias('Delta Time (s)'),
            pl.lit(sr / 2).alias('Delta Freq (Hz)')
        ])
        return file_df

    def build_dataset(self, dataset: datasets.Dataset):
        """Format all data in raven format.

        Args:
            ds (HF Dataset): Data to format.
        """
        # We don't need audio data, so disable decode
        dataset = dataset.cast_column(
            "audio", datasets.Audio(sampling_rate=self.sr, decode=False))

        df = dataset.to_polars()

        self.raven_ds = df.group_by(
            "audio").map_groups(self.default_template_annotation_style)

    def save(self, folder_path):
        """Save the Raven txt annotation files.

        Args:
            folder_path: Path to top level folder containing audio data
                audio data may be in many folder,
                idea is we use the filepath from
                the dataset to find it. This is the highest level path needed.
                Ideally the txt data is with your wav files,
                since raven opens both together
                and expects them in the same folder.
        """
        if self.raven_ds is None:
            raise ValueError("run build_dataset frist")

        if self.dry_run:
            print("Would create: ", folder_path)
        elif folder_path != "":
            os.makedirs(folder_path, exist_ok=True)

        n_groups = self.raven_ds['path'].n_unique()
        grouped = self.raven_ds.group_by('path')
        for file, file_df in tqdm.tqdm(
                grouped,
                desc="writing text files",
                total=n_groups
        ):
            file = file[0] if isinstance(file, tuple) else file

            base_name = os.path.splitext(os.path.basename(file))[0]
            output_file = os.path.join(folder_path, f"{base_name}.txt")

            if self.dry_run:
                print("Would create: ", output_file)
            else:
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                file_df = file_df.select(self.header)
                file_df.write_csv(output_file, separator="\t")


if __name__ == "__main__":
    # ===============================================================
    # Uncomment out the section below and fill in your classes as mapped
    # from ints, and the path to the output .pkl that results from
    # running inference.py on your data.
    # ===============================================================

    '''
    class_list = ['your', 'classes']

    import pickle
    import numpy as np
    from pathlib import Path

    path = '/path/to/inference/result/.pkl'
    with open(path, 'rb') as file:
        data = pickle.load(file)
    # labelstudio expects non-int values as predictions, map to labels
    data['labels'] = [
        class_list[i] for i in np.argmax(data['pred'], axis=1)
    ]
    ds = datasets.Dataset.from_dict({
        'audio': data['audio'],
        'labels': data['labels'],
        'offset': [item['offset'] for item in data['audio']],
        'duration': [item['duration'] for item in data['audio']]
    })
    '''

    # ===============================================================
    # Below is a fake dataset creation for demo purposes only.
    # In practice you would load your dataset from the saves in
    # whoot_model_training. Comment this out if using the above format.
    # ===============================================================

    ds = datasets.load_from_disk(
        "whoot_model_training/data/buow_data/cache/metadata.hf")["valid"]

    # ===============================================================
    # You may need to change the duration, sample rate, and whether or
    # not you want it to be a dry run or actually write the results.
    raven_ds = RavenDataset(ds, duration=3, sr=32_000, dry_run=False)

    # Raven wants audio and annotation files together
    PARENT_AUDIO_FOLDER = "/path/to/audio"
    raven_ds.save(PARENT_AUDIO_FOLDER)
