"""Permutation testing for Birdnet embedding buowset.

Usage:
    python3 permutation_test.py -meta /path/to/fold/metadata.csv
        -embeds /path/to/dir/with/birdnet/embeddings/
"""
import argparse
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
from tqdm import tqdm
from embed_to_df_birdnet import obtain_birdnet_embeddings, merge_dfs
from make_svm import get_binary_classes, make_x_and_y


def permutation_test(meta, embeds):
    """Conduct permutation test.

    Conducts a 100 iteration permutation test for whatever
    current data split is defined in make_x_and_y.

    Args:
        meta (str): Path to metadata with fold info.
        embeds (str): Path to birdnet embeddings.
    """
    data = pd.read_csv(meta, index_col=0)
    embed_dict = obtain_birdnet_embeddings(embeds)
    df_merged = merge_dfs(data, embed_dict)
    dataset = get_binary_classes(df_merged)
    permutated_accuracies = []
    permutation_iters = 100
    x_train, y_train, x_test, y_test = make_x_and_y(dataset)
    for _ in tqdm(range(permutation_iters), desc='Test Progress'):
        np.random.shuffle(y_train)
        np.random.shuffle(y_test)
        svm = SVC(class_weight='balanced', probability=True)
        svm.fit(x_train, y_train)
        y_pred_default = svm.predict(x_test)
        permutated_accuracies.append(accuracy_score(y_pred_default,
                                                    y_test)*100)
    print(f"Average permutated accuracy is: {np.mean(permutated_accuracies)}")


def main(meta, embeds):
    """Execute main script.

    Runs the permutation test function.

    Args:
        meta (str): Path to metadata with fold info.
        embeds (str): Path to birdnet embeddings.
    """
    permutation_test(meta, embeds)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(
        description='Input Directory Path'
    )
    PARSER.add_argument('-meta', type=str,
                        help='Path to fold metadata')
    PARSER.add_argument('-embeds', type=str,
                        help='Path to directory with embeddings files')
    ARGS = PARSER.parse_args()
    main(ARGS.meta, ARGS.embeds)
