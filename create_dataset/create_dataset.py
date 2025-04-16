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
    python3 create_dataset.py /path/to/human/labeled.csv
    /path/to/parent/dir/of/wavs/ /path/to/desired/output/dir/

"""
from create_segments import setup_logger, get_paths, create_segments
from create_segments import create_noise_segments, create_csv
from filter_labels import filter_labels_2017, filter_labels_2018
import argparse
import pandas as pd

def main():
    """
    """
    # parse the inputs
    '''parser = argparse.ArgumentParser(
        description='Input Directory Path'
    )
    parser.add_argument('-labels', type=str,
                        help='Path to human labeled csv')
    parser.add_argument('-wav_dir', type=str,
                        help='Path to directory containing wav files.')
    parser.add_argument('-output_dir', type=str,
                        help='Path to desired directory for segments.')
    parser.add_argument('-l', '--lengthen', type=int, default=0,
                        help='ms of padding for front and end of detection segment')
    parser.add_argument('-e', '--equalize', type=int,
                        help='each detection segment and noise segment will be the same length, not zero padded')
    args = parser.parse_args()
    main(args.labels, args.wav_dir, args.output_dir)'''
    wav_dir = "/mnt/buow/Acoustic_Recordings/2017-2018/2017/Otay/"
    labels = "/mnt/buow/Acoustic_Recordings/2017-2018/Results/Otay/2017/all.csv"
    output_dir = "/home/katiegarwood/create_dataset/"
    if output dir exists
        good, if not make
    if labels exist, good
        if not tell user
    if wav dir exists
        if not tell user
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
        #new_buow_rows = create_segments(wav, filtered_labels, output_dir)
        '''# get the number of labeled  detections for that wav
        num = num rows in new_rows
        # create same number of noise segments from the same wav file randomly
        new_noise_rows = create_noise_segments(wav, filtered_labels, num, output_dir)
        # combine the buow and noise annotations created
        new_rows = new_buow_rows + new_noise_rows
        # add the annotations to the csv of metadata for the dataset
        create_csv(new_rows)

        logging.info(f"Added " {int(new_rows)*2} "new segments from {wav}")'''


main()
