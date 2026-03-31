"""Split buowset into stratified k-folds.

Groups detections from the same site into 'groups'
and then determines the overall class distribution and
the class distribution for each 'group'. It allocates
all the groups to a 'fold' in a way where the folds
are roughly the same class distribution as the overall
dataset.

This script assumes xeno canto data and/or sagemic data may
be what is being grouped. If the generate_metadata script was
used to create the metadata from either of those souces,
this script will work to generate folds. You just have
to change the output filename and the names/amount of classes
to look for.

Usage:
    python3 strat_k_folds.py /path/to/metadata.csv
"""
import argparse
import pandas as pd
import numpy as np


from k_fold_split_copy import solve


def create_strat_folds(df):  # pylint: disable-msg=too-many-locals
    """Create grouped stratified k-folds.

    Args:
        df (pd.Dataframe): The metadata csv from when the dataset was created.

    Returns:
        pd.DataFrame: The same metadata but with labels as ints and a new fold
            column to denote the fold that segment is apart of.
    """
    num_classes = 2
    original_df = df
    df['label'] = df['label'].replace('Cactus Wren', 0)
    df['label'] = df['label'].replace('cactus_wren', 0)
    df['label'] = df['label'].replace('Common Raven', 1)
    df['label'] = df['label'].replace('common_raven', 1)
    df['label'] = df['label'].replace("Cassin's Finch", 1)
    df['label'] = df['label'].replace('cassins_finch', 1)
    df['label'] = df['label'].replace('California Towhee', 1)
    df['label'] = df['label'].replace('california_towhee', 1)
    df['label'] = df['label'].replace("Yellow-rumped Warbler", 1)
    df['label'] = df['label'].replace('yellow-rumped_warbler', 1)
    df['label'] = df['label'].replace('Wrentit', 1)
    df['label'] = df['label'].replace('wrentit', 1)
    # group is the subset of the index which is the wav file they all come from
    grouped = df.groupby('grouping_param')
    group_names = []
    group_matrix = []
    for index, group in grouped:
        counts = np.zeros(num_classes, dtype=int)
        label_counts = group['label'].value_counts()
        for label, count in label_counts.items():
            counts[int(label)] = count
        group_matrix.append(counts)
        group_names.append(index)
    problem = np.array(group_matrix)
    solution = solve(problem, k=5, verbose=True)
    # the fold allocation for each 'group'
    print(f"solution {solution}")
    print(np.sum(problem, axis=0) / np.sum(problem))
    folds = [problem[solution == i] for i in range(5)]
    fold_percents = np.array(
        [np.sum(folds[i], axis=0) / np.sum(folds[i]) for i in range(5)]
    )
    # the % of each class in each fold
    print(f"Fold percents: {fold_percents}")
    print(folds)
    grouped_original = original_df.groupby('grouping_param')
    df_with_folds = pd.DataFrame()
    count = 0
    for i, group in grouped_original:
        group['fold'] = solution[count]
        df_with_folds = pd.concat([df_with_folds, group])
        count += 1
    return df_with_folds


def main(meta):
    """Execute main script.

    Args:
        meta (str): Path to metadata csv from creating the dataset.
    """
    df = pd.read_csv(meta, index_col=0)
    df_with_folds = create_strat_folds(df)
    df_with_folds.to_csv("5-fold_metadata.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Input Directory Path'
        )
    parser.add_argument('meta', type=str,
                        help='Path to metadata csv')
    args = parser.parse_args()
    main(args.meta)
