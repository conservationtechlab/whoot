# Labeling Audio Data in Label Studio

This pipeline intends to take audiodatasets from the inferance or the data_extractors of `whoot_model_training` and format projects in label studio to easily verify annotations. The following readme outlines best practices for creating and intergrating with Label Studio

## Install Instructions

In the root folder run the following to install depencies for data exporter

`pip install -e .[data_exporters]`

## Creating new project

1) Create a .env file with the following properties in the same folder that this readme is in

```
# Define the URL where Label Studio is accessible
LABEL_STUDIO_URL = 'HOSTNAME OF LABEL STUDIO INSTANCE'
# API key is available at the Account & Settings page in Label Studio UI
LABEL_STUDIO_API_KEY = 'INSERT YOUR API KEY'
LABEL_STUDIO_PROJECT_ID = 'PROJECT ID FROM URL OF PROJECT'
```

2) Create a new project in that label studio instance and upload the needed data to it

NOTE: Save the project_id from the URL of the project

If keeping the data local on the instance, try to keep the file structure the same as is the audio file from your ML machine. For example, if some dataset is located at `mnt/datasets/audio_dataset_cool/AB/1/audio.wav` then you may want to make the path on label studio something like `label_studio_path/audio_dataset_cool/AB/1/audio.wav` for the easiest intergrations. Otherwise some minor file changes will be needed.

3) Run the script to apply annotations, see demo.py in this folder
