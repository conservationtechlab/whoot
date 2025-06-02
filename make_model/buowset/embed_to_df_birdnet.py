"""Creating a standardized dataframe with embeddings.

As there are multiple types and formats that can produce embeddings,
to train an SVM we need a consistent format for the data to be injested.
This function can be called from main to save the dataframe to disk,
and can be called elsewhere to return these dataframes programatically.
It specifically handles embeddings from running birdnet.embeddings on
all audio files.

Usage:
    python3 embeddings_to_df.py -embeds /path/to/embeddings/
        -path /path/to/output.csv
"""
import glob
import os
import ntpath
import argparse
import pandas as pd


def obtain_birdnet_embeddings(embeds):
    """Create a dict dataframe with filename and embedding list

    Args:
        embeds (str): Path to directory where embeddings files are.

    Returns:
        embed_df (pd.Dateframe): A dictonary with the filename as the
                                 key and the list of floats (embeddings)
                                 as the value
    """
    embed_dict = {}
    text_files = glob.glob(os.path.join(embeds, "*.txt"))
    for embed in text_files:
        filename = ntpath.basename(embed)
        filename = filename.replace(".birdnet.embeddings.txt", ".wav")
        dfb = pd.read_csv(embed,
                          delimiter="[,\t]",
                          engine='python',
                          header=None)
        dfb_stripped = dfb.drop(dfb.columns[:2], axis=1)
        flattened = dfb_stripped.values.flatten()
        if len(flattened) > 1024:
            print(f"filename {filename} has extra lines. Trunicating")
            flattened = flattened[:1024]
        embed_dict[filename] = flattened

    embed_df = pd.DataFrame.from_dict(embed_dict, orient='index')
    embed_df.index.name = 'filename'
    embed_df.reset_index(inplace=True)

    return embed_df


def main(embeds, path):
    """Main script to create and save out embedding df.

    Args:
        embeds (str): Path to the embeddings folder/file info.

        path (str): Path to output embeddings dataframe.csv.
    """
    embed_df = obtain_birdnet_embeddings(embeds)

    embed_df.to_csv(path, index=False)
    print(f"Created dataframe file: {path}")


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(
        description='Input Directory Path'
    )
    PARSER.add_argument('-embeds', type=str,
                        help='Path to directory containing embedding info.')
    PARSER.add_argument('-path', type=str,
                        help='Path to output dataframe')
    ARGS = PARSER.parse_args()
    main(ARGS.embeds, ARGS.path)
