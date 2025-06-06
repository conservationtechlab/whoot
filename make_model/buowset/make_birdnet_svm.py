"""Create embedding dataframe as well as SVM

Create the embedding and metadata combined dataframe for Birdnet
style embeddings output and create a binary SVM.

Usage:
    python3 make_birdnet_svm.py -meta /path/to/metadata.csv
        -embed_path /path/to/birdnet/embeddings/
        [OPTIONAL] -model /path/to/saved/model.pkl
"""
import argparse
import pandas as pd

from embed_to_df_birdnet import obtain_birdnet_embeddings, merge_dfs
from make_svm import get_binary_classes, make_svm, save_out_model


def main(meta, embed_path, model):
    """Create svm from raw bridnet embeddings and metadata.

    Args:
        embed_path (str): Path to birdnet embeddings files.
        meta (str): Path to metadata containing fold and labels.
        model (str): Path to desired save location of result model.pkl.
    """
    metadata = pd.read_csv(meta, index_col=0)
    embed_dict = obtain_birdnet_embeddings(embed_path)
    df_merged = merge_dfs(metadata, embed_dict)
    dataset = get_binary_classes(df_merged)
    if model is None:
        make_svm(dataset)
    else:
        svm = make_svm(dataset)
        save_out_model(svm, model)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(
        description='Input Directory Path'
    )
    PARSER.add_argument('-meta', type=str,
                        help='Path to metadata with fold and label info.')
    PARSER.add_argument('-embed_path', type=str,
                        help='Path to directory containing embedding info.')
    PARSER.add_argument('-model', type=str, default=None,
                        help='Path to output dataframe as .pkl.')
    ARGS = PARSER.parse_args()
    main(ARGS.meta, ARGS.embed_path, ARGS.model)
