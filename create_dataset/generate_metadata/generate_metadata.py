"""Generate metadata for xeno canto data and sagemic data.

In order to generate training and testing datasets from data
aqcuired from xeno canto and our sagemics, we need to be
able to create metadata that we can then create grouped,
stratified folds from to reduce leakage in these datasets.

Xeno-Canto - This script assumes that you have a folder where you have
segments generated from birdnet from xeno canto files, and
each species has its own folder. Data is grouped by original file,
so the unique xeno canto file identifier ID from the filename is the
grouping parameter.

SageMic - Assumes you have a folder with each day of recordings from
the sagemic, and only the species of interest are in those folders.
Data is grouped by date, so the folder name is the grouping
parameter.

Usage:

    python3 generate_metadata.py -audio_files /path/to/folder/

"""
import argparse
import os
from pathlib import Path
import pandas as pd


def extract_label_sagemic(filename):
    """Extract just the label from the sagemic filename.

    Args:
        filename (string): File path of the audio file.

    Returns:
        string: Label extracted from the filename.
    """
    file = Path(filename).name
    label = file.split('_')[1]
    return label


def extract_grouping_xc(filename):
    """Extract zthe unique xeno canto ID from the filename.

    Args:
        filename (string): File path of the audio file.

    Returns:
        string: ID extracted from the filename.
    """
    file = Path(filename).name
    after_second_underscore = file.split('_')[2]
    result = after_second_underscore.split('-')[0]
    return result


def extract_parent_dir(filename):
    """Extract the parent directory.

    The parent directory gives us significant info for both
    xeno canto data and sagemic data.

    Args:
        filename (string): File path of the audio file.

    Returns:
        string: Parent dir extracted from the path.

    """
    path = Path(filename)

    return path.parent.name


def make_dataframe(audio_path):
    """Walk the directory to create a column with the filepath.

    Args:
        audio_path (string): Working directory for all audio.

    Returns:
        pandas.DataFrame: Dataframe with the segment_path as the
            only column.
    """
    file_paths = []
    for root, _, files in os.walk(audio_path):
        for file in files:
            full_path = os.path.abspath(os.path.join(root, file))
            file_paths.append(full_path)

    dataframe = pd.DataFrame(file_paths, columns=["segment_path"])

    return dataframe


def main(audio_path, output):
    """Create metadata dataframe.

    Creates dataframe with segment paths, and fills the label and
    grouping parameter column based on either sagemic or xeno canto
    data format.

    Args:
        audio_path (string): Working directory for all audio files.
        output (string): Filename of the output metadata csv.
    """
    meta = make_dataframe(audio_path)
    # uncomment for sagemic metadata
    # meta['grouping_param'] = meta['segment_path'].apply(extract_parent_dir)
    # meta['label'] = meta['segment_path'].apply(extract_label_sagemic)
    # uncomment for xc metadata
    meta['grouping_param'] = meta['segment_path'].apply(extract_grouping_xc)
    meta['label'] = meta['segment_path'].apply(extract_parent_dir)
    meta.to_csv(output, index=False)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(
        description='Input Directory Path'
    )
    PARSER.add_argument('-audio_path', type=str,
                        help='Path to directory containing wav files.')
    PARSER.add_argument('-output', type=str,
                        help='Path to output metadata.csv')
    ARGS = PARSER.parse_args()
    main(ARGS.audio_path, ARGS.output)
