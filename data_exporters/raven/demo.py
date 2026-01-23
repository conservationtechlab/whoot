"""Demo for using Label Studio Exporter with a sample dataset.

Run from project root after saving a cache of buowset data
"""

import os
import random
from dotenv import load_dotenv
from label_studio import LabelStudioSetup
import datasets
import polars as pl
from datasets.utils.logging import disable_progress_bar, enable_progress_bar
import tqdm

class RavenDataset():
    def __init__(self, ds: datasets.Dataset, duration=3, sr:int = 32_000, label_col="labels", dry_run=True):
        self.ds = ds
        self.raven_ds = None
        self.sr = sr
        self.label_col = label_col
        self.duration = duration
        self.build_dataset(ds, duration=duration, sr=sr, label_col=label_col)
        self.header = [
            'Selection','View',"Begin Time (s)","End Time (s)","Low Freq (Hz)",'High Freq (Hz)','Delta Time (s)','Delta Freq (Hz)', "Label"
        ]
        self.dry_run = dry_run

    def default_template_annotation_style(self, file_df: pl.DataFrame):
        sr = self.sr

        offsets = file_df['offset'] if "offset" in file_df.columns else pl.lit(0)
        durations = file_df['duration'] if "duration" in file_df.columns else pl.lit(self.duration)

        file_df = file_df.unnest("audio")
        file_df = file_df.with_columns([
            pl.lit(1).alias('Selection'),
            pl.lit('Spectrogram 1').alias('View'),
            # TODO Okay this needs to be better so it can show the species name and whatnot
            pl.col(self.label_col).alias('Label').cast(pl.List(pl.String)).list.join(","),
            offsets.alias('Begin Time (s)'),
            (offsets + durations).alias('End Time (s)'),
            pl.lit(0).alias('Low Freq (Hz)'),
            pl.lit(sr/2).alias('High Freq (Hz)'),
            durations.alias('Delta Time (s)'),
            pl.lit(sr / 2).alias('Delta Freq (Hz)')
        ])
        return file_df

    def build_dataset(self, ds, duration=3, sr=32_000, label_col="labels"):
        self.sr = sr
        self.label_col = label_col
        self.duration = duration

        # We don't need audio data, so disable decode
        ds = ds.cast_column(
            "audio", datasets.Audio(sampling_rate=sr, decode=False))
        
        # if type(ds["train"].features["labels"]) == datasets.Sequence:
        #     names = ds["train"].features["labels"].feature.names
        #     ClassLabels = datasets.ClassLabel(num_classes=len(names), names=names)
        
        df = ds.to_polars()
        
        self.raven_ds = df.group_by(
            "audio").map_groups(self.default_template_annotation_style)

        

    def save(self, folder_path):
        """

        Args:
            parent_folder_path: Path to top level folder containing audio data
                audio data may be in many folder, idea is we use the filepath from
                the dataset to find it. This is the highest level path needed
                Ideally the txt data is with your wav files, since raven opens both together
                and expects them in the same folder
        """
        if self.raven_ds is None:
            raise ValueError("run build_dataset frist")

        if self.dry_run:
            print("Would create: ", folder_path)
        elif folder_path != "":
            os.makedirs(folder_path, exist_ok=True)

        n_groups = self.raven_ds['filepath'].n_unique()
        grouped = self.raven_ds.group_by('filepath')
        for file, file_df in tqdm.tqdm(grouped, desc="writing text files", total=n_groups):
            file = file[0] if isinstance(file, tuple) else file
            
            if file[0] == "/":
                file = file[1:]
            output_file = os.path.join(folder_path, f"{file.split(os.path.extsep)[0]}.txt")
            
            if self.dry_run:
                print("Would create: ", output_file)
            else:
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                file_df.select(self.header).write_csv(output_file, separator="\t")

if __name__ == "__main__":

    # ===============================================================
    # below is a fake dataset creation for demo purposes only
    # In practice you would load your dataset from the saves in
    # whoot_model_training

    # This cache gets created when running the buowset_extractor in whoot_model_training
    ds = datasets.load_from_disk("data/burrowing_owl_dataset/cache/metadata.hf")["valid"]
    # ===============================================================

    raven_ds = RavenDataset(ds, sr=16_000, dry_run=False)

    # Raven wants audio and annotation files together
    parent_audio_folder = ""
    raven_ds.save(parent_audio_folder)
