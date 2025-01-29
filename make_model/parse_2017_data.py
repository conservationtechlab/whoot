"""Standardize the human labels.

Each sound file of interest has a corresponding csv created that contains
labels for each 3 second chunk in the file, where 'yes' is marked on
that time chunk if the human label contains a vocalization in that time
frame.

Example:

    $ python parse_2017_data.py /path/to/human_labels.csv \
      /path/to/directory/of/wavs/ /path/to/directory/output/

"""

import argparse
import os
import pandas as pd
import librosa


def main(labels, wavs, output):
    """Create human labeled dataframes.

    Main script to create csvs of human labeled data for each
    wav file of interest.

    Args:
        labels (str): The path to human labeled csv.
        wavs (str): The path to all audio files.
        output (str): The path to directory where each csv will
            output (1 for each wav).

    """
    os.makedirs(output, exist_ok=True)

    scored_data = pd.read_csv(labels)

    for audio_file in os.listdir(wavs):
        if audio_file.endswith('.wav'):
            audio_path = os.path.join(wavs, audio_file)

            try:
                time_series, sample_rate = librosa.load(audio_path, sr=None)
                audio_duration = librosa.get_duration(y=time_series,
                                                      sr=sample_rate)
            except Exception as err:
                print(f"Error processing {audio_file}: {err}")
                continue

            total_chunks = int(audio_duration // 3) + 1
            chunks_data = {
                'Chunk Start': [i * 3 for i in range(total_chunks)],
                'Chunk End': [(i + 1) * 3 for i in range(total_chunks)],
                'Label': ['no'] * total_chunks
            }
            chunks_df = pd.DataFrame(chunks_data)

            filtered_data = scored_data[scored_data['IN FILE'] == audio_file]
            for _, row in filtered_data.iterrows():
                if row['TOP1MATCH'] != 'null':
                    start_time = float(row['OFFSET'])
                    end_time = start_time + float(row['DURATION'])

                    for i in range(len(chunks_df)):
                        chunk_start = chunks_df.loc[i, 'Chunk Start']
                        chunk_end = chunks_df.loc[i, 'Chunk End']
                        if (start_time < chunk_end and end_time > chunk_start):
                            chunks_df.loc[i, 'Label'] = 'yes'

            output_file = os.path.join(
                output, f'{os.path.splitext(audio_file)[0]}_chunks.csv'
            )
            chunks_df.to_csv(output_file, index=False)
            print(f"Processed {audio_file} -> {output_file}")

    print("Processing complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Input Directory Path'
        )
    parser.add_argument('labels', type=str,
                        help='Path to human labeled csv')
    parser.add_argument('wavs', type=str,
                        help='Path to all wav files that have been labeled')
    parser.add_argument('output', type=str,
                        help='Path to desired directory for output csvs')
    args = parser.parse_args()
    main(args.labels, args.wavs, args.output)
