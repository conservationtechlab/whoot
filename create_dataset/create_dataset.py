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
/
    python3 create_dataset.py -labels /path/to/human/labeled.csv
    -wav_dir /path/to/parent/dir/of/wavs/ -output_dir /path/to/desired/output/dir/
    -class_list /path/to/classes.txt

"""
from create_segments import setup_logger, get_paths, create_segments
from create_segments import create_noise_segments, create_csv
from filter_labels import filter_labels_2017, filter_labels_2018
import argparse
import pandas as pd
import ntpath
import os

def create_dataset(labels, wav_dir, output_dir, class_list):
    """
    """
    # parse the inputs
    '''if output dir exists
        good, if not make
    if labels exist, good
        if not tell user
    if wav dir exists
        if not tell user'''
    out_file = ntpath.dirname(output_dir)
    result_file = os.path.join(out_file, "metadata.csv")
    if os.path.exists(result_file):
        all_data = pd.read_csv(result_file)
    else:
        all_data = pd.DataFrame()
    # walk dir to list paths to each original wav file
    wav_file_paths = get_paths(wav_dir)
    # open human label file
    labels = pd.read_csv(labels)
    #iterate through each individual original wav
    if "2017" in labels['DATE'].iloc[0]:
        use_2017 = True
    elif "2018" in labels['DATE'].iloc[0]:
        use_2017 = False
    for wav in wav_file_paths:
        # check which label format to select parsing method
        # create dataframe of only the labels that correspond to the wav
        if use_2017 == True:
            filtered_labels = filter_labels_2017(wav, labels)
        elif use_2017 == False:
            filtered_labels = filter_labels_2018(wav, labels)
        # output the labeled segments and return the dataframe of annotations
        new_buow_rows = create_segments(wav, filtered_labels, output_dir, class_list)
        # create same number of noise segments from the same wav file randomly
        all_buow_rows = create_noise_segments(wav, new_buow_rows, output_dir)
        # add the annotations to the csv of metadata for the dataset
        
        all_data = pd.concat([all_data, all_buow_rows], ignore_index=True)
        print(all_data)

        print(f"Added  {len(all_buow_rows)} new segments from {wav}")
    all_data.to_csv(result_file)
    print(f"Created results: {result_file}")

def main(labels, wav_dir, output_dir, class_list):
    """
    """
    create_dataset(labels, wav_dir, output_dir, class_list)

if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description='Input Directory Path'
    )
    parser.add_argument('-labels', type=str,
                        help='Path to human labeled csv')
    parser.add_argument('-wav_dir', type=str,
                        help='Path to directory containing wav files.')
    parser.add_argument('-output_dir', type=str,
                        help='Path to desired directory for segments.')
    parser.add_argument('-class_list', type=str,
                        help='Path to txt file of list of labeled classes')
    #parser.add_argument('-l', '--lengthen', type=int, default=0,
        #                help='ms of padding for front and end of detection segment')
    # parser.add_argument('-e', '--equalize', type=int,
      #                   help='each detection segment and noise segment will be the same length, not zero padded')
    args = parser.parse_args()
    main(args.labels, args.wav_dir, args.output_dir, args.class_list)
