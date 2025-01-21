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
import numpy as np


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
        stripped_filename = filename.strip("_chunks.csv")
        for birdnet in os.listdir(embeddings):
            stripped_birdnet = birdnet.strip(".birdnet.embeddings.txt")
            if stripped_birdnet == stripped_filename:
                birdnet_path = os.path.join(embeddings, birdnet)
                dfb = pd.read_csv(birdnet_path, delimiter=",", header=None)
                dfb_stripped = dfb.drop(dfb.columns[:2], axis=1)
                dfb_stripped.columns = [f"feature_{i}" for i in range(1, len(dfb_stripped.columns) + 1)]
                df_stripped  = compare_dfs(df, dfb_stripped)
                combined_df = pd.concat([df_stripped, dfb_stripped], axis=1)
                output_filename = stripped_filename + "_labeled_embeddings.csv"
                output_path = os.path.join(output, output_filename)
                combined_df.to_csv(output_path, index=False)
                print(f"Labeled embeddings created for: {output_path}") 
            else:
                continue

def compare_dfs(df, dfb):
    """Ensure embeddings and labels have same number of rows.

    Args:
        df: the labeled dataframe
        dfb: the embeddings

    Returns:
        proper_df: labels with correct number of rows

    """
    if abs(len(df) - len(dfb)) > 1:
        print(abs(len(df) - len(dfb)))
        raise ValueError("Dfs have a difference greater than 1 row")

    if len(df) > len(dfb):
        df_stripped = df.iloc[:-1]

    return df_stripped


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



 
