"""Standardize Birdnet Output.

This script takes a master csv containing the aggregated birdnet output
text files and creates an expanded csv containing  a "no" label
for each chunk of time that birdnet did not detect a vocalization. The
3-second chunks where birdnet made a detection will be marked with a "yes".

Example:

    $ python normalize_birdnet_output.py \
      /path/to/aggregated_birdnet_output.csv /path/to/birdnet_labeled.csv

"""

import argparse
import pandas as pd
import numpy as np


def main(aggr_birdnet, birdnet_labeled):
    """Organize and expand birdnet labels.

    Main function to take birdnet labels and create a
    dataframe that has time chunks for the whole audio file
    duration and labels the detection periods with "yes".

    Args:
        aggr_birdnet (str): Path to aggregated Birdnet analysis file.
        birdnet_labeled (str): Path to desired output csv.

    """
    ml_output = pd.read_csv(aggr_birdnet)

    filtered_data = ml_output[ml_output['Common Name'] == 'burowl']

    # Optional if Birdnet analysis is continuous and not aggregated.
    filtered_data = filtered_data.apply(adjust_time, axis=1)

    # Total duration of the sound file(s) that Birdnet analyzed.
    total_duration = 10800
    all_intervals = pd.DataFrame({
        'Begin Time (s)': np.arange(0, total_duration, 3),
        'End Time (s)': np.arange(3, total_duration + 3, 3),
    })
    all_intervals['Label'] = 'no'

    for _, row in filtered_data.iterrows():
        start = row['Begin Time (s)']
        end = row['End Time (s)']
        mask = (
            all_intervals['Begin Time (s)'] >= start
        ) & (all_intervals['End Time (s)'] <= end)
        all_intervals.loc[mask, 'Label'] = 'yes'

    all_intervals.to_csv(birdnet_labeled, index=False)


def adjust_time(row):
    """Create continuous timestamps.

    Function to standardize the timestamps for the aggregated
    birdnet input file. This is because this script was designed
    assuming that the analysis needed to be aggregated from split
    wav files from the same larger audio recording. This function
    ensures that if you split up your sound file in smaller bits
    to be analyzed by birdnet and aggregate their output, the time
    chunks will represent the entire sound file in order and not
    as separate audio files.

    Args:
        row (pandas.Series): The current row in the dataframe.

    Returns:
        pandas.Series: The time adjusted row in the dataframe.

    """
    chunk_number = int(row['File Name'].split('output_')[1].split('.')[0])
    offset = chunk_number * 60
    row['Begin Time (s)'] += offset
    row['End Time (s)'] += offset
    return row


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Input Directory Path'
        )
    parser.add_argument('aggr_birdnet',
                        type=str,
                        help='File path to aggregated birdnet raw output')
    parser.add_argument('birdnet_labeled',
                        type=str,
                        help='Result csv with adjusted birdnet results')
    args = parser.parse_args()
    main(args.aggr_birdnet, args.birdnet_labeled)
