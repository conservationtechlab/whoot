"""Zero padding human labeled detections

In order to obtain birdnet embeddings, each sample
must be 3s long at least. The human labeled detections are
often shorter than 3s. This script adds silence to make
samples shorter than 3s, reach 3s. Birdnet needs the files
to be hardcoded paths in order to later obtain the embeddings
so we must save out these new files, and we also duplicate the
original so we're essentially creating a copy of the dataset
but with no segments less than 3s.

Usage: python3 zero_pad_detections.py -i /path/to/dir/wavs/
    -o /path/to/new/dataset/
"""
import argparse
import os
from pydub import AudioSegment


def pad_segments(path, output):
    """Pad segments with silence to reach 3s

    For segments shorter than 3 seconds, we add silence to the
    end to reach 3s in length minimum.

    Args:

        path (str): Path to all of the audio segments.

        output (str): Path to desired output for all segments
                      now lengthened to 3s minimum
    """
    for file in os.listdir(path):
        filepath = os.path.join(path, file)
        pad_ms = 3000  # Add here the fix length you want (in milliseconds)
        audio = AudioSegment.from_wav(filepath)
        if len(audio) < pad_ms:
            silence = AudioSegment.silent(duration=pad_ms-len(audio)+1)
            padded = audio + silence  # Adding silence after the audio
            full_path = os.path.join(output, file)
            padded.export(full_path, format='wav')
        else:
            full_path = os.path.join(output, file)
            audio.export(full_path, format='wav')


def main(path, output):
    """Main function

    Runs pad segments.

    Args:

        path (str): Path to all of the audio segments.

        output (str): Path to desired output for all segments
                      now lengthened to 3s minimum
    """
    pad_segments(path, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Input Directory Path'
    )
    parser.add_argument('-path', type=str,
                        help='Path to dataset audio clips')
    parser.add_argument('-o', type=str,
                        help='Path to desired output directory for all clips')
    args = parser.parse_args()
    main(args.path, args.o)

# TODO: There are some buow vocalizations longer than 3s.
# Currently, the make_svm just trunicates the birdnet embeddings longer than
# one 3-second feature detection, but we should handle that here.
