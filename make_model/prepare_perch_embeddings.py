'''
Convert Perch Embedding Output to standard embeddings .pkl
for easy training of various models

This script processes the outputs of the perch embedding
scripts and converts to a .pkl dataframe that stores
filename, embedding, and related metadata

Usage: python prepare_perch_embeddings \
           /path/to/sqlite_dir \
           /path/to/metadata_file \
           /path/to/output_dir \
           embeddings_description

Arguments:
    sqlite_dir (str): path to directory that contains
        hoplite.sqlite & usearch.index
    metadata_path (str): path to metadata file with labels
        and fold information
    outout_dir (str): path to directory to store output
        pkl
    embeddings_description (str) :description of set of embeddings
        for file naming purposes

Outputs:
    <embeddings_description>_perch_embeddings.pkl

'''


import os
import argparse
import pandas as pd
from perch_hoplite.db import sqlite_usearch_impl


def prepare_perch_embeddings(sqlite_dir,
                             metadata_path,
                             output_dir,
                             embeddings_description):
    '''
    converts raw perch embeddings (from sqlite database ) into standard
    dataframe format for SVM.

    Args:
        sqlite_dir (str): path to directory that contains
            hoplite.sqlite $ usearch.index
        metadata_path (str): path to metadata file
        output_dir (str): path to directory to store
            output .pkl file
        embeddings_description (str): description of set of embeddings
            for file naming purposes

    Returns:
        None
    '''

    # load embeddings database
    db = sqlite_usearch_impl.SQLiteUsearchDB.create(sqlite_dir)

    # load dataset metadata
    metadata = pd.read_csv(metadata_path, index_col=0)

    embeddings_data = []

    n_embeddings = db.count_embeddings()

    for i in range(n_embeddings):

        file_name = db.get_embedding_source(i+1).source_id
        embedding = db.get_embedding(i+1)

        base_dict = {'segment': file_name,
                     'embedding': embedding}

        embeddings_data.append(base_dict)

    embeddings_df = pd.DataFrame(embeddings_data)
    merged_df = pd.merge(embeddings_df, metadata, on='segment')
    merged_df = merged_df.drop('segment_duration_s', axis=1)
    merged_df = merged_df[['segment', 'label', 'fold', 'embedding']]

    output_filename = os.path.join(output_dir, f'{embeddings_description}_perch_embeddings.pkl')
    merged_df.to_pickle(output_filename)

    print(f'Embeddings saved at:\n\t{output_filename}')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Input Perch Embeddings sqlite database and output directory')

    parser.add_argument('sqlite_dir', type=str,
                        help='Path to directory that contains '
                             'hoplite.sqlite and usearch.index')
    parser.add_argument('metadata_path', type=str,
                        help='Path to metadata file')
    parser.add_argument('output_dir', type=str,
                        help='Directory for output file')
    parser.add_argument('embeddings_description', type=str,
                        help='Name of embeddings group')

    args = parser.parse_args()
    prepare_perch_embeddings(args.sqlite_dir, args.metadata_path, args.output_dir, args.embeddings_description)
