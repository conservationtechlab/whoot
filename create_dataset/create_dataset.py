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
noise will be fixed and consistent. The user of the dataset may choose
to pad the labeled detections if they need consistent length segments.

Usage:
    python3 create_dataset.py -labels /path/to/human/labeled.csv
    -wav_dir /path/to/parent/dir/of/wavs/
    -output_dir /path/to/desired/output/dir/
    -class_list /path/to/classes.txt

"""
import argparse
import ntpath
import os
import pandas as pd
from create_segments import get_paths, create_segments
from create_segments import create_noise_segments
from filter_labels import filter_labels_2017, filter_labels_2018


def create_dataset(labels, wav_dir, output_dir, class_list):
    """Creates labeled and non labeled segments and metadata.

    Creates segments based on human labeled data of a detection,
    and then creates an equal number of randomized 'non-detection'
    segments at fixed length. It cretaes a uuid for each segment
    and spits out a metadata file that matches the segment to its
    label, original wav file, relative start time to original wav,
    and duration.

    Args:
        labels (str): Path to label file.

        wav_dir (str): Path to original wav segments of audio.

        output_dir (str): Path to where the segments and metadata
                          will go.

        class_list (str): Path to file containing the classes
                          seen in the human labels file that you
                          want to create segments for.
    """
    # parse the inputs
    out_file = ntpath.dirname(output_dir)
    result_file = os.path.join(out_file, "metadata.csv")
    if os.path.exists(result_file):
        all_data = pd.read_csv(result_file, index_col=0)
    else:
        all_data = pd.DataFrame()
    # walk dir to list paths to each original wav file
    wav_file_paths = get_paths(wav_dir)
    # open human label file
    labels = pd.read_csv(labels)
    use_2017 = None
    # iterate through each individual original wav
    if "2017" in labels['DATE'].iloc[0]:
        use_2017 = True
    elif "2018" in labels['DATE'].iloc[0]:
        use_2017 = False
    wav_files = []
    num_samples = []
    for wav in wav_file_paths:
        # check which label format to select parsing method
        # create dataframe of only the labels that correspond to the wav
        if use_2017:
            filtered_labels = filter_labels_2017(wav,
                                                 labels)
        else:
            filtered_labels = filter_labels_2018(wav,
                                                 labels)
        # output the labeled segments and return the dataframe of annotations
        new_buow_rows = create_segments(wav,
                                        filtered_labels,
                                        output_dir,
                                        class_list)
        # create same number of noise segments from the same wav file randomly
        all_buow_rows = create_noise_segments(wav,
                                              new_buow_rows,
                                              output_dir)
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


def main(labels, wav_dir, output_dir, class_list):
    """Main script to run create dataset.

    Args:
        labels (str): Path to label file.

        wav_dir (str): Path to original wav segments of audio.

        output_dir (str): Path to where the segments and metadata
                          will go.

        class_list (str): Path to file containing the classes
                          seen in the human labels file that you
                          want to create segments for.
    """
    create_dataset(labels, wav_dir, output_dir, class_list)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(
        description='Input Directory Path'
    )
    PARSER.add_argument('-labels', type=str,
                        help='Path to human labeled csv')
    PARSER.add_argument('-wav_dir', type=str,
                        help='Path to directory containing wav files.')
    PARSER.add_argument('-output_dir', type=str,
                        help='Path to desired directory for segments.')
    PARSER.add_argument('-class_list', type=str,
                        help='Path to txt file of list of labeled classes')
    ARGS = PARSER.parse_args()
    main(ARGS.labels, ARGS.wav_dir, ARGS.output_dir, ARGS.class_list)
