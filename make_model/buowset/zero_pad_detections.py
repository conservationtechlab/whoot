"""Zero padding human labeled detections.

In order to obtain birdnet embeddings, each sample
must be 3s long at least. The human labeled detections are
often shorter than 3s. This script adds silence to make
samples shorter than 3s, reach 3s. Birdnet needs the files
to be hardcoded paths in order to later obtain the embeddings
so we must save out these new files, and we also duplicate the
original so we're essentially creating a copy of the dataset
but with no segments less than 3s.

Usage: python3 zero_pad_detections.py -path /path/to/dir/wavs/
    -output /path/to/new/dataset/ -length 3000 -randomize

Omitting -length and -randomize will default to 3000ms and NON
random padding (padding added to end of sample)
"""
import argparse
import os
import random
from pydub import AudioSegment


def pad_segments(path, output, length, randomize):
    """Pad segments with silence to reach desired duration.

    For segments shorter than min duration, we add silence to the
    end to reach the desired length.

    Args:

        path (str): Path to all of the audio segments.

        output (str): Path to desired output for all segments
                      now lengthened.

        length (int): Desired minimum duration of padded segments in ms.
                      Default 3000ms.

        randomize (bool): Flag for if location of padded silence is randomized
                          within the length of the segment.
    """
    for file in os.listdir(path):
        filepath = os.path.join(path, file)
        audio = AudioSegment.from_wav(filepath)
        if len(audio) < length:
            if randomize:
                max_begin_silence = length - len(audio)
                begin_silence = random.uniform(0.0, max_begin_silence)
                end_silence = length - (len(audio) + begin_silence)
                begin_silence = AudioSegment.silent(duration=begin_silence)
                end_silence = AudioSegment.silent(duration=end_silence)
                padded = begin_silence + audio + end_silence
            else:
                silence = AudioSegment.silent(duration=length-len(audio)+1)
                padded = audio + silence  # Adding silence after the audio
            full_path = os.path.join(output, file)
            padded.export(full_path, format='wav')
        else:
            full_path = os.path.join(output, file)
            audio.export(full_path, format='wav')


def main(path, output, length, randomize):
    """Main function.

    Runs pad segments.

    Args:

        path (str): Path to all of the audio segments.

        output (str): Path to desired output for all segments
                      now lengthened to desired duration.

        length (int): Minimum duration of the resulting audio
                      segments, in milliseconds.

        randomize (bool): Flag for if the location of the padded
                          silence is randomized within the length
                          of the segment.
    """
    pad_segments(path, output, length, randomize)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(
        description='Input Directory Path'
    )
    PARSER.add_argument('-path', type=str,
                        help='Path to dataset audio clips')
    PARSER.add_argument('-output', type=str,
                        help='Path to desired output directory for all clips.')
    PARSER.add_argument('-length', type=int, default=3000,
                        help='Minimum length(ms) of the clips, default 3000.')
    PARSER.add_argument('-randomize', action='store_true',
                        help='Randomize location of the audio amidst silence.')
    ARGS = PARSER.parse_args()
    main(ARGS.path, ARGS.output, ARGS.length, ARGS.randomize)

# TODO: There are some buow vocalizations longer than 3s.
# Currently, the make_svm just trunicates the birdnet embeddings longer than
# one 3-second feature detection, but we should handle that here.
