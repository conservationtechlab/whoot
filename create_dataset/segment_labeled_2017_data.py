"""Create human labeled audio segments.

Using a CSV with human labels across a large dataset, we can
find the segments in the audio files that correspond to a
burrowing owl call as labeled by a human labeler. We can then
segment these audio chunks into a folder so that we can use
them to easily train other models. We can also do the same
for the rest of the data to obtain segments with no bird
call labels, to provide another class in the same domain
as our bird vocalizations. As there are significantly more
negatives than positives, we can choose if we'd like to get
the same number output or select a higher or lower amount.

Example:

    $ python segment_labeled_2017_data.py /path/to/human_labels.csv \
      /path/to/directory/of/wavs/ /path/to/directory/output/

"""

import argparse
import os
import pandas as pd
import librosa
from pydub import AudioSegment


def create_bird_segments(labels, wavs, output):
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
    output = output + "bird_sounds/"
    os.makedirs(output, exist_ok=True)

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
            bird_sound = AudioSegment.from_wav(audio_path)
            segment_index = 0
            for _, row in filtered_data.iterrows():
                if row['TOP1MATCH'] != 'null':
                    start_time = float(row['OFFSET'])
                    end_time = start_time + float(row['DURATION'])

                    for i in range(len(chunks_df)):
                        chunk_start = chunks_df.loc[i, 'Chunk Start']
                        chunk_end = chunks_df.loc[i, 'Chunk End']
                        if (start_time < chunk_end and end_time > chunk_start):
                            chunk_start = chunk_start * 1000
                            chunk_end = chunk_end * 1000
                            segment = bird_sound[chunk_start:chunk_end]
                            output_file = os.path.join(
                                output, f'{os.path.splitext(audio_file)[0]}_segment_{segment_index}.wav'
                            )
                            segment.export(output_file, format='wav')
                            segment_index += 1

    print("Processing complete!")

def create_no_bird_segments(labels, wavs, output):
    """Create no bird call audio segments.

    """
    os.makedirs(output, exist_ok=True)

    scored_data = pd.read_csv(labels)
    output = output + "no_bird_sounds/"
    os.makedirs(output, exist_ok=True)

    for audio_file in os.listdir(wavs):
        if audio_file.endswith('.wav'):
            audio_path = os.path.join(wavs, audio_file)

            try:
                time_series, sample_rate = librosa.load(audio_path, sr=None)
                audio_duration = librosa.get_duration(y=time_series, sr=sample_rate)
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
                        if start_time < chunk_end and end_time > chunk_start:
                            chunks_df.loc[i, 'Label'] = 'bird'

            bird_sound = AudioSegment.from_wav(audio_path)
            segment_index = 0
            for i in range(len(chunks_df)):
                if chunks_df.loc[i, 'Label'] == 'no':
                    chunk_start = chunks_df.loc[i, 'Chunk Start'] * 1000
                    chunk_end = chunks_df.loc[i, 'Chunk End'] * 1000
                    segment = bird_sound[chunk_start:chunk_end]

                    output_file = os.path.join(
                        output, f'{os.path.splitext(audio_file)[0]}_nobird_segment_{segment_index}.wav'
                    )
                    segment.export(output_file, format='wav')
                    segment_index += 1

    print("Processing complete!")

def main(labels, wavs, output):
    """Run main script

    """
    create_bird_segments(labels, wavs, output)
    create_no_bird_segments(labels, wavs, output)


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
