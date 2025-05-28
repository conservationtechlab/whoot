'''
Convert Perch Embedding Output to standard embeddings .pkl
for easy training of various models

This script processes the outputs of the perch embedding
scripts and converts to a .pkl dataframe that stores
filename, embedding, and related metadata

Usage: python prepare_perch_embeddings \
           /path/to/db/dir \
           /path/to/metadata/file \
           /path/to/output/dir \
           embeddings_description

Arguments:
    database_directory (str): path to directory that contains
        hoplite.sqlite & usearch.index
    outout_directory (str): path to directory to store output
        csv

Outputs:
    <description>_perch_embeddings.pkl

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
    runs main script
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

        #embedding_dict = {f'{j}': val for j, val in enumerate(embedding)}

        #full_row = {**base_dict, **embedding_dict}

        embeddings_data.append(base_dict)

    embeddings_df = pd.DataFrame(embeddings_data)
    merged_df = pd.merge(embeddings_df, metadata, on='segment')

    output_filename = os.path.join(output_dir, f'{embeddings_description}_perch_embeddings.pkl')
    merged_df.to_pickle(output_filename)

#    merged_df.to_csv(csv_filename, index=False)

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
