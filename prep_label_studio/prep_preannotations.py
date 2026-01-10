from label_studio_sdk import LabelStudio, label_interface
from label_studio import LabelStudioSetup
from datasets import Dataset
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-np.array(x)))

def annotation_formatter_row(row, class_list):
        return label_interface.objects.ob.AnnotationValue(result={
            {
                "value": {
                    "start": 0,
                    "end": 1,
                    "labels": (np.array(class_list)[sigmoid(row["pred"]) > 0.5]).tolist()
                },
                "from_name": "labels",
                "to_name": "audio",
                "type": "labels",
                "origin": "manual"
            },
        })





# TODO Add regions level metadata to this... need that... thats gonna get confusing damn.
def label_studio_format_row(row):
    return {
      "data": {
        "ref_id": 456,
        "meta_info": {
          "filename": row["audio"]["path"],
        }
      },
      "Annotations": {
        "from_name": "species_name",
        "to_name": "audio",
        "type": "choices",
        "readonly": False,
        "hidden": False,
        "value": {
          "choices": (np.array(class_list)[sigmoid(row["pred"]) > 0.5]).tolist()
        }
      }
    }

def hs_ds_to_label_studio(ds, threshold=0.5):
    return ds.map(label_studio_format_row)


test = hs_ds_to_label_studio(ds)
test


def load_annotations_to_ls(datasets: Dataset, ls_wrapper: LabelStudioSetup):
    """Loads annotations from a Dataset into Label Studio.

    Args:
        datasets (Dataset): The dataset containing annotations.
        ls_wrapper (LabelStudioSetup): The Label Studio setup instance.
    """

    tasks = []
    for idx, row in datasets.iterrows():
        data = {
            "data": {
                "audio": row["audio_path"]
            },
            "annotations": [
                {
                    "result": row["annotations"]
                }
            ]
        }
        project.import_tasks([data])