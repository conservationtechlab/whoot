"""Split buowset into stratified k-folds.

Groups detections from the same site into 'groups'
and then determines the overall class distribution and
the class distribution for each 'group'. It allocates
all the groups to a 'fold' in a way where the folds
are roughly the same class distribution as the overall
dataset.

Usage:
    python3 strat_k_folds.py /path/to/metadata.csv /path/to/site_list.txt
"""
import argparse
import pandas as pd
import numpy as np


from k_fold_split_copy import solve


def create_strat_folds(df, site_ids):
    """Create grouped stratified k-folds.

    Args:
        df (pd.Dataframe): The metadata csv from when the dataset was created.
        site_ids (list): Site names (found in original file path) to group by.

    Returns:
        pd.DataFrame: The same metadata but with labels as ints and a new fold
            column to denote the fold that segment is apart of.
    """
    num_classes = 6
    for index, row in df.iterrows():
        for site_id in site_ids:
            if site_id in row['original_path']:
                df.loc[index, 'site_id'] = site_id
    print(df.head)
    original_df = df
    df['label'] = df['label'].replace('cluck', 0)
    df['label'] = df['label'].replace('coocoo', 1)
    df['label'] = df['label'].replace('twitter', 2)
    df['label'] = df['label'].replace('alarm', 3)
    df['label'] = df['label'].replace('chick begging', 4)
    df['label'] = df['label'].replace('no_buow', 5)
    # group is the subset of the index which is the wav file they all come from
    grouped = df.groupby('site_id')
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
    grouped_original = original_df.groupby('site_id')
    df_with_folds = pd.DataFrame()
    count = 0
    for i, group in grouped_original:
        group['fold'] = solution[count]
        df_with_folds = pd.concat([df_with_folds, group], ignore_index=True)
        count += 1
    return df_with_folds


def main(meta, sites):
    """Execute main script.

    Args:
        meta (str): Path to metadata csv from creating the dataset.
        sites (str): Path to sites to group by.
    """
    df = pd.read_csv(meta, index_col=0)
    with open(sites, 'r', encoding='utf-8') as file:
        site_ids = [line.strip() for line in file.readlines()]
    df_with_folds = create_strat_folds(df, site_ids)
    df_with_folds.to_csv("5-fold_meta.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Input Directory Path'
        )
    parser.add_argument('meta', type=str,
                        help='Path to metadata csv')
    parser.add_argument('sites', type=str,
                        help='Path to site list')
    args = parser.parse_args()
    main(args.meta, args.sites)
