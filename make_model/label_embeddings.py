"""Script to merge embeddings with labeled data

In order to make use of the birdnet embeddings for each sound
file to train an svm (or any model), we want to ensure each
labeled chunk contains the embedding information as well.
You will need to have run all your audio through embeddings.py
to obtain the embeddings for each of the sound files. This script
then takes both the embeddings and the labeled output csvs from
running parse_2017_data.py to create 1 csv that contains the human
ground truth label as well as columns for each of the 1024 features
per 3 second chunk.

python label_embeddings.py /path/to/output/ /path/to/birdnet_embeddings/ /path/to/desired/outputs/
"""
import argparse
import


def main(human_labels, embeddings, output):
    """Main script

    Args:
        human_labels:
        embeddings:
        output:

    """
    



 
