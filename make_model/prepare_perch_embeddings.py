'''
Convert Perch Embedding Output to standard embeddings .csv
for easy training of various models

This script processes the outputs of the perch embedding
scripts and converst to a usable .csv file that stores
filename and embedding

Usage: python prepare_perch_embeddings \
           /path/to/db/dir \
           /path/to/output/dir \
           embeddings_description

Arguments:
    database_directory (str): path to directory that contains
        hoplite.sqlite & usearch.index
    outout_directory (str): path to directory to store output
        csv

Outputs:
    <description>_perch_embeddings.csv

'''


import os
import argparse
import pandas as pd
from perch_hoplite.db import sqlite_usearch_impl


def main(sqlite_dir, output_dir, embeddings_description):
    '''
    runs main script
    '''

    # load database
    db = sqlite_usearch_impl.SQLiteUsearchDB.create(sqlite_dir)

    master_data = []

    n_embeddings = db.count_embeddings()

    for i in range(n_embeddings):

        file_name = db.get_embedding_source(i+1).source_id

        base_dict = {'filename': file_name}

        embedding = db.get_embedding(i+1)
        embedding_dict = {f'{j}': val for j, val in enumerate(embedding)}

        full_row = {**base_dict, **embedding_dict}

        master_data.append(full_row)

    master_df = pd.DataFrame(master_data)
    csv_filename = os.path.join(output_dir, f'{embeddings_description}_perch_embeddings.csv')
    master_df.to_csv(csv_filename, index=False)

    print('Complete!')
    print(f'Saved at:\n\t{csv_filename}')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Input Perch Embeddings sqlite database and output directory')

    parser.add_argument('sqlite_dir', type=str,
                        help='Path to directory that contains '
                             'hoplite.sqlite and usearch.index')
    parser.add_argument('output_dir', type=str,
                        help='Directory for output file')
    parser.add_argument('embeddings_description', type=str,
                        help='Name of embeddings group')

    args = parser.parse_args()
    main(args.sqlite_dir, args.output_dir, args.embeddings_description)
