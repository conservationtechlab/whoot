"""Demo for using Label Studio Exporter with a sample dataset.

Run from project root after saving a cache of buowset data
"""

import os
import random
from dotenv import load_dotenv
from label_studio import LabelStudioSetup
import datasets
from datasets.utils.logging import disable_progress_bar, enable_progress_bar
import tqdm

class RavenDataset():
    def __init__(self, ds: datasets.Dataset, duration=3, sr:int = 32_000, label_col="labels", dry_run=True):
        self.ds = ds
        self.raven_ds = None
        self.build_dataset(ds, duration=duration, sr=sr, label_col=label_col)
        self.header = [
            'Selection','View',"Begin Time (s)","End Time (s)","Selection","Low Freq (Hz)",'High Freq (Hz)','Delta Time (s)','Delta Freq (Hz)', "Label"
        ]
        self.dry_run = dry_run

    def default_template_annotation_style(
            self,
            offset,
            duration,
            label,
            sr,
            row
    ):
        out = {
            'Selection': 1,
            'View': 'Spectrogram 1',
            "Begin Time (s)": offset,
            "End Time (s)": offset + duration,
            "Selection": 1,
            "Low Freq (Hz)": 0,
            'High Freq (Hz)': sr / 2,
            'Delta Time (s)': duration,
            'Delta Freq (Hz)': sr / 2,
            "Label": label
        }
        return out

    def build_dataset(self, ds, duration=3, sr=32_000, label_col="labels"):
        raven_ds = {}

        # We don't need audio data, so disable decode
        ds = ds.cast_column(
            "audio", datasets.Audio(sampling_rate=sr, decode=False))

        print("Collecting files")
        disable_progress_bar()
        files = list(set([row["audio"]["path"] for row in  tqdm.tqdm(ds)]))
        print("Format raven files")
        for i in tqdm.tqdm(
            range(len(files)),
            # desc=f"Updating tasks in project: {self.current_project.title}"
        ):
            
            file_ds = ds.filter(lambda x: x['audio']['path'] == files[i])
            file_ds = file_ds.map(
                lambda x: self.default_template_annotation_style(
                    x["audio"]["offset"] if "offset" in x["audio"] else 0,
                    x["audio"]["duration"] if "offset" in x["audio"] else duration,
                    x[label_col], 
                    sr,
                    x
                )
            )
            

            raven_ds[files[i]]= file_ds

        enable_progress_bar()
        print("Raven dataset created, run save() to save files")
        self.raven_ds = raven_ds

    def save(self, folder_path):
        """

        Args:
            folder_path: Path to folder to put txt files
                Ideally with your wav files, since raven opens both together
        """
        if self.raven_ds is None:
            raise ValueError("run build_dataset frist")

        if self.dry_run:
            print("Would create: ", folder_path)
        else:
            os.makedirs(folder_path, exist_ok=True)

        for file in self.raven_ds:
            file_ds = self.raven_ds[file]
            
            if file[0] == "/":
                file = file[1:]
            output_file = os.path.join(folder_path, f"{file.split(os.path.extsep)[0]}.txt")
            
            if self.dry_run:
                print("Would create: ", output_file)
            else:
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                file_ds.to_csv(output_file, sep="\t", columns=self.header)

if __name__ == "__main__":

    # ===============================================================
    # below is a fake dataset creation for demo purposes only
    # In practice you would load your dataset from the saves in
    # whoot_model_training

    # This cache gets created when running the buowset_extractor in whoot_model_training
    ds = datasets.load_from_disk("data/burrowing_owl_dataset/cache/metadata.hf")["valid"]
    # ===============================================================

    # UPLOAD DATASET TO LABEL STUDIO
    raven_ds = RavenDataset(ds)

    # Raven wants audio and annotation files together
    audio_folder = "data/burrowing_owl_dataset/audio"
    raven_ds.save(audio_folder)
