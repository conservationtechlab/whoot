"""Merge embeddings with labeled data.

In order to make use of the birdnet embeddings for each sound
file to train an svm (or any model), we want to ensure each
labeled chunk contains the embedding information as well.
You will need to have run all your audio through embeddings.py
to obtain the embeddings for each of the sound files. This script
then takes both the embeddings and the labeled output csvs from
running parse_2017_data.py to create 1 csv that contains the human
ground truth label as well as columns for each of the 1024 features
per 3 second chunk.

Example:

    $ python label_embeddings.py /path/to/output/ \
      /path/to/birdnet_embeddings/ /path/to/desired/outputs/

"""

import argparse
import os
import pandas as pd


def main(human_labels, embeddings, output):
    """Merge dataframes.

    Args:
        human_labels (str): The path to the human labeled csv.
        embeddings (str): The path to the Birdnet embeddings files.
        output (str): The path to desired output directory.

    """
    for filename in os.listdir(human_labels):
        file_path = os.path.join(human_labels, filename)
        label_df = pd.read_csv(file_path)
        stripped_filename = filename.strip("_chunks.csv")

        for birdnet in os.listdir(embeddings):
            stripped_birdnet = birdnet.strip(".birdnet.embeddings.txt")

            if stripped_birdnet == stripped_filename:
                birdnet_path = os.path.join(embeddings, birdnet)
                dfb = pd.read_csv(birdnet_path,
                                  delimiter="[,\t]",
                                  engine='python',
                                  header=None)
                dfb_stripped = dfb.drop(dfb.columns[:2], axis=1)
                dfb_stripped.columns = [
                    f"feature_{i}" for i in range(
                        1, len(dfb_stripped.columns) + 1)]
                df_stripped = compare_dfs(label_df, dfb_stripped)
                combined_df = pd.concat([df_stripped, dfb_stripped], axis=1)
                output_filename = stripped_filename + "_labeled_embeddings.csv"
                output_path = os.path.join(output, output_filename)
                combined_df.to_csv(output_path, index=False)
                print(f"Labeled embeddings created for: {output_path}")

            else:
                continue


def compare_dfs(label_df, dfb):
    """Ensure same number of rows.

    The human labeled data may contain one more row than the
    Birdnet embeddings because it will ignore the end of a file
    if there is not enough time for a full 3-second final chunk.
    This will remove that last human labeled line as there will
    be no embedding for it. It will throw an error if it's off
    by more than one row because that should never be the case.

    Args:
        df (pandas.Dataframe): The human labeled dataframe.
        dfb (pandas.Dataframe): The embeddings dataframe.

    Returns:
        proper_df (pandas.Dataframe: The labeled dataframe with
            correct number of rows.

    """
    if abs(len(label_df) - len(dfb)) > 1:
        raise ValueError("Dfs have a difference greater than 1 row")

    if len(label_df) > len(dfb):
        df_stripped = label_df.iloc[:-1]
    else:
        df_stripped = label_df

    return df_stripped


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Input and output paths'
        )
    parser.add_argument('labels', type=str,
                        help='Directory path to human labeled csvs.')
    parser.add_argument('embeddings', type=str,
                        help='Directory path to birdnet embeddings.')
    parser.add_argument('output', type=str,
                        help='Directory path to desired output csvs.')
    args = parser.parse_args()
    main(args.labels, args.embeddings, args.output)
