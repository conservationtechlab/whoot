"""Demo for using Label Studio Exporter with a sample dataset."""

import os
import random
from dotenv import load_dotenv
from label_studio import LabelStudioSetup
import datasets
import tqdm

class RavenDataset():
    def __init__(self, ds: datasets.Dataset, duration=3, sr:int = 32_000, label_col="labels"):
        self.ds = ds
        self.raven_ds = None
        self.build_dataset(ds, duration=duration, sr=sr, label_col=label_col)
        self.header = [
            'Selection','View',"Begin Time (s)","End Time (s)","Selection","Low Freq (Hz)",'High Freq (Hz)','Delta Time (s)','Delta Freq (Hz)', "Label"
        ]

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
        files = list(set([ds["audio"][i]["path"] for i in range(len(ds["audio"]))]))
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

            print(file_ds[0])

            raven_ds[files[i]]= file_ds

        self.raven_ds = raven_ds

    def save(self, folder_path):
        """

        Args:
            folder_path: Path to folder to put txt files
                Ideally with your wav files, since raven opens both together
        """
        if self.raven_ds is None:
            raise ValueError("run build_dataset frist")

        os.makedirs(folder_path, exist_ok=True)

        for file in self.raven_ds:
            file_ds = self.raven_ds[file]
            
            if file[0] == "/":
                file = file[1:]
            output_file = os.path.join(folder_path, f"{file.split(os.path.extsep)[0]}.txt")
            print(output_file, os.path.dirname(output_file))
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            file_ds.to_csv(output_file, sep="\t", columns=self.header)

if __name__ == "__main__":

    # ===============================================================
    # below is a fake dataset creation for demo purposes only
    # In practice you would load your dataset from the saves in
    # whoot_model_training
    load_dotenv()

    # SELECT A PROJECT FROM LABEL STUDIO
    # FIND ID IN URL OF PROJECT
    PROJECT_ID = int(os.getenv("LABEL_STUDIO_PROJECT_ID"))
    ls_setup = LabelStudioSetup(
        current_project=PROJECT_ID
    )
    # ADD DEFAULT TEMPLATE TO LABEL STUDIO
    ls_setup.apply_custom_template("template.xml")

    # HOW TO GET AUDIO FILES TO REVIEW
    # Note this is not a perfect process as
    # diffrences between label studio and your dataset may exist
    file_meta = ls_setup.get_files(ls_file_parent='data/local-files/?d=data1/')

    # Make sure your file names align to label studio files

    class_list = ['cluck', 'coocoo',
                  'twitter', 'alarm', 'chick begging', 'no_buow']

    ds = datasets.Dataset.from_dict({
        "audio": random.choices(file_meta["files"], k=len(file_meta["files"])),
        "labels": random.choices(
            class_list, k=len(file_meta["files"])
        )
    })

    ds = ds.cast_column(
        "audio", datasets.Audio(sampling_rate=16000, decode=False))
    # ===============================================================

    # UPLOAD DATASET TO LABEL STUDIO
    raven_ds = RavenDataset(ds)
    raven_ds.save("raven_test")
