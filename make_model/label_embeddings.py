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
import pandas as pd
import os


def main(human_labels, embeddings, output):
    """Main script

    Args:
        human_labels:
        embeddings:
        output:

    """
    for filename in os.listdir(human_labels):
        file_path = os.path.join(human_labels, filename)
        df = pd.read_csv(file_path)
        df.add 1024 empty columns
        filename = strip _chunks.csv
        for birdnet in os.listdir(embeddings):
            birdnet_path = os.path.join(embeddings, birdnet)
            dfb = pd.read(birdnet_path)
            strip first 2 columns
            if check num rows -1 equals num rows in df:
                fill the 1024 column info from dfb into df-
                save the file to new df in output%03.csv
                print(f"Labeled embeddings created for: {output_filename}") 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Input Directory Path'
        )
    parser.add_argument('labels', type=str,
                        help='Directory path to human labeled csvs')
    parser.add_argument('embeddings', type=str,
                        help='Directory path to birdnet embeddings')
    parser.add_argument('output', type=str,
                        help='Directory path to desired output csvs')
    args = parser.parse_args()
    main(args.labels, args.embeddings, args.output)



 
