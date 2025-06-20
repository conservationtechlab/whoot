Toolkit for training Machine Learning Classification Models over audio dataset

Key inspiration is https://github.com/UCSD-E4E/pyha-analyzer-2.0/tree/main. This repo differs in that it uses a traditional training pipeline rather than the Hugging Face Trainer. Hugging face trainer abstracts the training code, which should be explicit for this toolkit. 


# Install

To set up environment for model training:

1) run steps 1 - 3 of the installation instructions in `whoot/README.md`
2) For step 4, specifically run `pip install -e .[model_training, cu128/cpu]`

# Running

0) Add your Comet-ML API to your local environment. See 
1) Create a copy of the config found in `configs/config.yml` and fill it out with your dataset
2) Edit train.py to set up training for your dataset. If you are using a new dataset which an extractor does not exist for, contact code authors. 
3) run `python train.py path/to/your/config/file.yml`
