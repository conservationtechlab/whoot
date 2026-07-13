"""Create dataset of burrowing owl vocalizations and noise.

This script will parse through 2017 and 2018 human labeled
burrowing owl data. It will create a folder with segments of
labeled detections, and an equal number of noise samples from
the same wav files. It will create a CSV with metadata associated
with the segments. The metadata will include the UUID of the segment,
the label, the original filepath of the original wav the segment came
from, the path to the segment, and the start and end time of the
labeled detection relative to the original wav file. The labeled
segments will be the duration of the label, and the duration of the
noise will be fixed and consistent. Adding -we (window expansion)
parameter will expand the window to a default 3s for detections,
evenly expanding on both sides around the sample. -randomize
parameter will randomly add window expansion to the default 3s
but the detection can be anywhere within those 3s.

Usage (from /whoot):
    python3 -m create_dataset.create_dataset -config /path/to/config.yaml

"""
import argparse
import ntpath
import os
import yaml
import pandas as pd
from whoot import get_paths, create_segments
from whoot import create_noise_segments
from whoot import default_filter, custom_filter


def create_dataset(config):
    """Creates labeled and non labeled segments and metadata.

    Creates segments based on human labeled data of a detection,
    and then creates an equal number of randomized 'non-detection'
    segments at fixed length. It cretaes a uuid for each segment
    and spits out a metadata file that matches the segment to its
    label, original wav file, relative start time to original wav,
    and duration.

    Args:
        config (dict): Dictionary of config values.
    """
    # parse the inputs
    out_file = ntpath.dirname(config["output_dir"])
    result_file = os.path.join(out_file, "metadata.csv")
    if os.path.exists(result_file):
        all_data = pd.read_csv(result_file, index_col=0)
    else:
        all_data = pd.DataFrame()
    # walk dir to list paths to each original wav file
    wav_file_paths = get_paths(config["wav_dir"])
    # open human label file
    labels = pd.read_csv(config["labels"])
    # iterate through each individual original wav
    wav_files = []
    num_samples = []
    for wav in wav_file_paths:
        # check which label format to select parsing method
        # create dataframe of only the labels that correspond to the wav
        if config["default_filter"] == True:
            filtered_labels = default_filter(wav, labels, config["filepath"])
        elif config["default_filter"] == False:
            filtered_labels = custom_filter(wav,
                                            labels)
        # output the labeled segments and return the dataframe of annotations
        new_buow_rows = create_segments(wav, filtered_labels, config)
        print(f"new buow rows: {new_buow_rows}")
        # create same number of noise segments from the same wav file randomly
        all_buow_rows = create_noise_segments(wav,
                                              new_buow_rows,
                                              config["output_dir"])
        # add the annotations to the csv of metadata for the dataset
        if not all_buow_rows.empty:
            wavv = str(wav)
            wav_files.append(wavv)
            num_samples.append(len(all_buow_rows))
        all_data = pd.concat([all_data, all_buow_rows], ignore_index=True)
        print("printing concated data")
        print(all_data)

    all_data.index = all_data.index.astype(int)
    all_data.to_csv(result_file)
    intt = 0
    for wavs in wav_files:
        print(f"{wavs} had {num_samples[intt]} including noise segments")
        intt += 1
    print(f"Created results: {result_file}")


def main(config_file):
    """Main script to run create dataset.

    Args:
        config_file (str): Path to config file.
    """
    with open(config_file, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    create_dataset(config)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(
        description='Input Directory Path'
    )
    PARSER.add_argument('-config', type=str,
                        help='Path to config file.')
    ARGS = PARSER.parse_args()
    main(ARGS.config)
