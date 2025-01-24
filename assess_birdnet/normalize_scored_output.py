"""Adjust human labeled data

This script takes the human labeled data and will output a
dataframe containing time chunks for your specific sound file
of interest and labels for each chunk depending on what
detections were marked by the human labelers.

Example:
    $ python normalize_scored_output.py /path/to/human_labeled.csv /path/to/output_dataframe.csv

# pylint: disable=line-too-long
"""
import argparse
import pandas as pd


def main(labels, adjusted_labels):
    """Organize and expand human labeled data
    Main function to create a dataframe with the whole
    duration of the audio file of interest represented in
    time chunks labeled 'no' or 'yes' if the human labels
    marked a vocalization in that specific time chunk.

    Args:
        labels (str): The path to the human labeled data.
        adjusted_labels (str): The resulting csv dataframe with
        labels for each 3 second chunk based on the human labels.

    """
    scored_data = pd.read_csv(labels)

    # need to insert wav file of interest, cannot handle multiple at once
    file_of_interest = '20170421_180000.wav'
    filtered_data = scored_data[scored_data['IN FILE'] == file_of_interest]

    # time length of audio file of interest
    audio_file_duration = 10800

    total_chunks = audio_file_duration // 3
    chunks_data = {
        'Chunk Start': [i*3 for i in range(total_chunks)],
        'Chunk End': [(i+1)*3 for i in range(total_chunks)],
        'Label': ['no'] * total_chunks
    }
    chunks_df = pd.DataFrame(chunks_data)
    filtered_data.apply(lambda row: mark_intervals(row, chunks_df),
                        axis=1)

    chunks_df.to_csv(adjusted_labels, index=False)
    print(f"File {adjusted_labels} created successfully.")


def mark_intervals(row, chunks_df):
    """Labeling detections
    Function to relabel the row in the dataframe
    to yes if the human labels marked a vocalization
    at that point.

    Args:
        row (pandas.Series object): The current row in the human labeled
        data that matches the audio file of interest.
        chunks_df (pandas.Dataframe object): The new unlabeled dataframe.

    """
    start_time = float(row['OFFSET'])
    end_time = start_time + float(row['DURATION'])
    start_chunk = int(start_time // 3)
    end_chunk = int(end_time // 3)

    if row['TOP1MATCH'] != 'null':
        chunks_df.loc[start_chunk:end_chunk, 'Label'] = 'yes'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Input Directory Path'
        )
    parser.add_argument('labels',
                        type=str,
                        help='File path to human labeled raw output')
    parser.add_argument('adjusted_labels',
                        type=str,
                        help='Result csv with adjusted human labels')
    args = parser.parse_args()
    main(args.labels, args.adjusted_labels)
