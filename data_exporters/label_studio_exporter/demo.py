
"""Demo for using Label Studio Exporter with a sample dataset."""

import os
import random
from dotenv import load_dotenv
from label_studio import LabelStudioSetup
import datasets

if __name__ == "__main__":

    load_dotenv()

    ls_file_parent='label-studio/local/data1/panda_inference/?d=data1/'

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
    file_meta = ls_setup.get_files(ls_file_parent)

    # Make sure your file names align to label studio files

    # ===============================================================
    # below is an example for loading in your inference results

    '''
    # class list must not be ints
    # ensure template.xml matches this list
    class_list = ['your', 'classes']

    import pickle
    import numpy as np
    from pathlib import Path

    path = '/path/to/result/pickle/from/inference.pkl'
    with open(path, 'rb') as file:
        data = pickle.load(file)
    # labelstudio expects non-int values as predictions, map to labels
    data['labels'] = [
        class_list[i] for i in np.argmax(data['pred'], axis=1)
    ]
    # audio path must match labelstudio path which is a string
    for item in data['audio']:
        item['path'] = Path(item['path']).name
        for file in file_meta['files']:
            if item['path'] in file:
                item['path'] = str(file)

    ds = datasets.Dataset.from_dict({
        'audio': data['audio'],
        'labels': data['labels']
    })
    '''
    # ===============================================================

    # ===============================================================
    # below is a fake dataset creation for demo purposes only
    # In practice you would load your dataset from the saves in
    # whoot_model_training using the above example

    class_list = ['cluck', 'coocoo',
                  'twitter', 'alarm', 'chick begging', 'no_buow']

    ds = datasets.Dataset.from_dict({
        "audio": file_meta["files"],
        "labels": random.choices(
            class_list, k=len(file_meta["files"])
        )
    })

    ds = ds.cast_column(
        "audio", datasets.Audio(sampling_rate=16000, decode=False))
    # ===============================================================

    # UPLOAD DATASET TO LABEL STUDIO
    ls_setup.update_tasks_in_ls(
        ds,
        ls_file_parent,
        is_model_prediction=True
    )
