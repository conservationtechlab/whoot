'''
Convert Perch Embedding Output to usabale .csvs

This script processes the outputs of the perch embedding
scripts and converst to a usable .csv file of all embeddings
with labels to use for training a binary SVM classifier

Usage: python make_perch_svm_dataset.py /path/to/db/dir label

Arguments:
    database_directory (str): path to directory that contains
        hoplite.sqlite & usearch.index
    label (str): label for all embeddings in hoplite.sqlite

Outputs:
    label_embeddings_forSVM.csv

'''


import argparse
import sqlite3
import pandas as pd
import sys
import os
from perch_hoplite.db import sqlite_usearch_impl


def split_base_segment(filename):
    '''separates source file base and segment #
    example: audio_segment_3.wav -> audio, 3

    Args:
        filename: segment file name

    Returns:
        source file base name & segment #
    '''

    base, _ = os.path.splitext(filename)
    return base.split('_segment_')

def get_start_stop(seg_id):
    ''' calculates start stop timestamp in s
        from segment number

    Args:
        seg_id (int): segment number of audio chunk

    Returns:
        start s, stop s

    '''
    seg_id = int(seg_id)
    return (seg_id*3, (seg_id+1)*3)


def main(sqlite_dir, label, output_dir):
    '''
    runs main script
    '''

    # load database
    db = sqlite_usearch_impl.SQLiteUsearchDB.create(sqlite_dir)

    master_data = []

    n_embeddings = db.count_embeddings()

    for i in range(n_embeddings):

        file_name = db.get_embedding_source(i+1).source_id
        base_name, segment_id = split_base_segment(file_name)
        start, stop = get_start_stop(segment_id)
        base_dict = {'start': start,
                     'stop': stop,
                     'label': label}

        embedding = db.get_embedding(i+1)
        embedding_dict = {f'feature_{j}': val for j, val in enumerate(embedding)}

        full_row = {**base_dict, **embedding_dict}

        master_data.append(full_row)


    master_df = pd.DataFrame(master_data)
    csv_filename = f'{output_dir}/{label}_embeddings_forSVM.csv'
    master_df.to_csv(csv_filename)

    print('Complete!')
    print(f'Saved at:\n\t{csv_filename}')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                  description='Input SQLite Direcotry and Label')

    parser.add_argument('sqlite_dir', type=str,
                        help='Path to directory that contains '
                             'hoplite.sqlite and usearch.index')
    parser.add_argument('label', type=str,
                        help='Label for all embeddings in given db')
    parser.add_argument('output_dir', type=str,
                        help='Directory for output file')

    args = parser.parse_args()
    main(args.sqlite_dir, args.label, args.output_dir)
