'''
Create Perch Embeddings Script

This script processes a directory of audio chunks (.wav files),
creates perch embeddings, and stores the results as a sqlite database

Usage:
    python make_perch_embeddings.py dataset_name path/to/directory/of/wavs
        path/to/desired/output/dir

Outputs:
    hoplite.sqlite
    usearch.index

Note:
    this code requires:
        python, version 3.10+
        numpy, version 1.2+
        tensorflow, version 2+
'''

import argparse
from etils import epath

from perch_hoplite.agile import colab_utils
from perch_hoplite.agile import embed
from perch_hoplite.agile import source_info
from perch_hoplite.db import interface

def create_embeddings(dataset_name, wavs, output):
    '''
    creates perch embeddings

    Args:
        dataset_name (str): name of dataset being embedded
        wavs (str): path to directory containing .wav audio segments
        output (str): path to directory for output files (SQLite DB)

    Returns:
        None
    '''

    dataset_base_path = wavs
    dataset_fileglob = '*.wav'
    db_path = output
    model_choice = 'perch_8'

    use_file_sharding = True

    audio_glob = source_info.AudioSourceConfig(
        dataset_name=dataset_name,
        base_path=dataset_base_path,
        file_glob=dataset_fileglob,
        min_audio_len_s=1.0,
        target_sample_rate_hz=-2,
        shard_len_s=60.0 if use_file_sharding else None,
    )

    configs = colab_utils.load_configs(
        source_info.AudioSources((audio_glob,)),
        db_path,
        model_config_key=model_choice,
        db_key='sqlite_usearch')

    # Initialize DB
    db = configs.db_config.load_db()
    num_embeddings = db.count_embeddings()
    print('Initialized DB located at ', configs.db_config.db_config.db_path)

    def drop_and_reload_db() -> interface.HopliteDBInterface:
        db_path = epath.Path(configs.db_config.db_config.db_path)
        for fp in db_path.glob('hoplite.sqlite*'):
            fp.unlink()
        (db_path / 'usearch.index').unlink()
        print('\n Deleted previous db at: ',
              configs.db_config.db_config.db_path)

    if num_embeddings > 0:
        print('Existing DB contains datasets: ', db.get_dataset_names())
        print('num embeddings: ', num_embeddings)
        print(f'This will permanently delete all {num_embeddings} '
              'embeddings from the existing database.\n')
        drop_and_reload_db()

    # Run embedding
    print(f'Embedding dataset: {audio_glob.dataset_name}')

    worker = embed.EmbedWorker(
        audio_sources=configs.audio_sources_config,
        db=db,
        model_config=configs.model_config)

    worker.process_all(target_dataset_name=audio_glob.dataset_name)

    print('\n\nEmbedding complete! \nTotal embeddings: ', db.count_embeddings())
    print(f'Embeddings dataset saved at: \n '
          f'\t{output}/hoplite.sqlite \n '
          f'\t{output}/usearch.index')

def main(dataset_name, wavs, output):
    '''
    run main script
    '''

    create_embeddings(dataset_name, wavs, output)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Input Directory Paths'
    )
    parser.add_argument('dataset_name', type=str,
                        help='Name of dataset to embed')
    parser.add_argument('wavs', type=str,
                        help='Path to labeled audio chunks. '
                             'All .wav files will be embedded')
    parser.add_argument('output', type=str,
                        help='Path to desired directory for output database')
    args = parser.parse_args()

    main(args.dataset_name, args.wavs, args.output)
