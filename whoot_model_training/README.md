Toolkit for training Machine Learning Classification Models over audio dataset

Key inspiration is https://github.com/UCSD-E4E/pyha-analyzer-2.0/tree/main. This repo differs in that it uses a traditional training pipeline rather than the Hugging Face Trainer. Hugging face trainer abstracts the training code, which should be explicit for this toolkit. 


# Install

To set up environment for model training:

1) run steps 1 - 3 of the installation instructions in `whoot/README.md`
2) For step 4, specifically run `pip install -e .[model-training, cpu]` for cpu training,  `pip install -e .[model-training, cu128]` for training on Nvidia GPUs

Note that you should check what is supported by CUDA on your machine. See developers if you need a different CUDA version 

# Running

0) Add your Comet-ML API to your local environment. See 
1) Create a copy of the config found in `configs/config.yml` and fill it out for your dataset. See the [config](#config) section
2) Edit train.py to set up training for your dataset. If you are using a new dataset which an extractor does not exist for, contact code authors. 
3) run `python train.py path/to/your/config/file.yml`

# Config

## Default Config Properties
The properties of `config.yml` are as follows:
### Data paths
`metadata_csv`: the path to the metadata file for your dataset.
`data_path`: Path to the highest level parent folder containing audio. Audio can be in a different path than the metadata!
`hf_cache_path`: cache for hugging face. This path will be automatically made as you run the script, this would be the location of where the new file should go

### Required Variables
`COMET_PROJECT_NAME`: "whoot", this is the project on comet-ml training will run on. 
`CUDA_VISIBLE_DEVICES`: "0" or "0,1", this controls how many GPUs the training uses.
`SUBPROJECT_NAME`: Some description to help filter which training this is used for, can be the task being done (multi_label_classification) or something else (fun_training_test)
`DATASET_NAME`: Name of the dataset being trained on, will be embedded on comet_ml to make searching easier

## Project Specific config information
### Buowset
The filenames in metadata_csv are the audio files found in `data_path`. 

`SUBPROJECT_NAME` is either "binary" or "mutlilabelClass"
`DATASET_NAME` is buowset0

# Repo Philosophy  

The most challenging issue with machine learning is the dataset. This training repo intends to make it easy to modularize parts of the training pipeline, and integrate them together, ideally regardless of the dataset. 

The pipeline works in 5 parts:
- Extractors: Extractors take in raw data and reformats it into `AudioDatasets`, apache-arrow data structures implemented via HuggingFace with common columns between any dataset. Every label is one_hot_encoded and treated as mutlilabel regardless of the problem. Audio filepaths as casted into [Audio columns](https://huggingface.co/docs/datasets/v3.6.0/en/package_reference/main_classes#datasets.Audio). Extractors are *unique for each dataset* but *uniform in the AudioDataset*. 

- Preprocessors: Online preprocessors take rows in `AudioDatasets` and output `ModelInputs`, formatted data specific to a given model. Preprocessors read AudioDatasets and translate it so the Model can read it

- Models: Models have defined `ModelInput` and `ModelOutput` formats. All ModelInputs and ModelOutputs have common data that they are required to have such that the `PyhaTrainer` can understand how to feed information to the Model, and how to read information from the model. All models implement their own loss functions and return a loss given labels. 

- Augmentations: TODO

- PyhaTrainer: With few exceptions unrelated to bioacoustic classifications, all PyTorch training code is the same. The HuggingFace Trainer and the extension PyhaTrainer handle most training scripts you will ever write. Why not use it and focus on model design, dataset preprocessing and cleaning. As long as the trainer knows how to feed data into a model (`AudioDatasets` and `Preprocessors`) and how to read it (`ModelOutputs`), then it will have no issues. 