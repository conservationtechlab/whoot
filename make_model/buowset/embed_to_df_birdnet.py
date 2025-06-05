"""Creating a standardized dataframe with embeddings.

As there are multiple types and formats that can produce embeddings,
to train an SVM we need a consistent format for the data to be injested.
This function can be called from main to save the dataframe to disk,
and can be called elsewhere to return these dataframes without saving to disk.
It specifically handles embeddings from running birdnet.embeddings on
all audio files. It joins the fold and label info from the metadata
file with the embeddings list and wav name of origin.

Usage:
   $  python3 embeddings_to_df.py -embeds /path/to/embeddings/
      -meta /path/to/metadata.csv -path /path/to/output.pkl
"""
import glob
import os
import ntpath
import argparse
import pandas as pd


def obtain_birdnet_embeddings(metadata, embeds):
    """Merge embeddings with fold and label info.

    Args:
        embeds (str): Path to directory where embeddings files are.

    Returns:
        pd.DateFrame: A dataframe with the embedding info,
            fold and label info in columns.
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
    embed_df.index.name = 'segment'
    df_merged = metadata.merge(embed_df, on='segment')
    df_merged = df_merged.drop(columns=['segment_duration_s'])

    return df_merged


def main(meta, embeds, path):
    """Main script to create and save out embedding df.

    Args:
        meta (str): Path to metadata file.
        embeds (str): Path to the embeddings folder/file info.
        path (str): Path to output embeddings dataframe.csv.
    """
    metadata = pd.read_csv(meta, index_col=0)
    df_merged = obtain_birdnet_embeddings(metadata, embeds)

    df_merged.to_pickle(path)

    print(f"Created dataframe file: {path}")


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(
        description='Input Directory Path'
    )
    PARSER.add_argument('-meta', type=str,
                        help='Path to metadata with fold and label info.')
    PARSER.add_argument('-embeds', type=str,
                        help='Path to directory containing embedding info.')
    PARSER.add_argument('-path', type=str,
                        help='Path to output dataframe as .pkl.')
    ARGS = PARSER.parse_args()
    main(ARGS.meta, ARGS.embeds, ARGS.path)
