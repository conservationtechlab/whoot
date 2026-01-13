"""Demo for using Label Studio Exporter with a sample dataset."""

from label_studio import LabelStudioSetup
from dotenv import load_dotenv
import os   

load_dotenv()


## SELECT A PROJECT FROM LABEL STUDIO
## FIND ID IN URL OF PROJECT
PROJECT_ID = int(os.getenv("LABEL_STUDIO_PROJECT_ID"))
ls_setup = LabelStudioSetup(
    current_project=PROJECT_ID
)
## ADD DEFAULT TEMPLATE TO LABEL STUDIO
ls_setup.apply_custom_template("template.xml")

## HOW TO GET AUDIO FILES TO REVIEW
# Note this is not a perfect process as diffrences between label studio and your dataset may exist
ls_setup.get_files(ls_file_parent='data/local-files/?d=data1/')

## TODO MAKE SURE YOUR DATASET AND LABEL STUDIO FILE PATHS ALIGN

#===============================================================
## below is a fake dataset creation for demo purposes only
## In practice you would load your dataset from the saves in whoot_model_training
import datasets
import random

class_list = ['cluck','coocoo', 'twitter', 'alarm', 'chick begging', 'no_buow']

ds = datasets.Dataset.from_dict({
    "audio": ls_setup.get_files(ls_file_parent='data/local-files/?d=data1/')["files"],
    "labels": random.choices(class_list, k=len(ls_setup.get_files(ls_file_parent='data/local-files/?d=data1/')["files"]))
})

ds = ds.cast_column("audio", datasets.Audio(sampling_rate=16000, decode=False))

#===============================================================



## UPLOAD DATASET TO LABEL STUDIO
ls_setup.update_tasks_in_ls(ds, ls_file_parent='data/local-files/?d=data1/', is_model_prediction=True)


