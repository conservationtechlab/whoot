"""Create segments of noisy audio from wavs.

This script uses the extract noise function to
calculate the average RMS of a given wav file,
and then creates 3 second segments where the
RMS peaked above the average. This main script
parses through a directory and sends each wav
file through the function. The extract_noise.yaml
is an example of the config file needed, copy
it and fill it out prior to running script.

Usage:

    python3 run_extract_noise.py
    -config /path/to/extract_noise_copy.yaml

"""
import argparse
import os
import yaml
from whoot.extract_noise import clip_loud_segments


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(
        description='Path to config file.'
    )
    PARSER.add_argument('-config', type=str,
                        help='Path to config.')
    ARGS = PARSER.parse_args()
    with open(ARGS.config, 'r', encoding='UTF-8') as f:
        config = yaml.safe_load(f)
    all_files = os.listdir(config['audio'])
    for file in all_files:
        try:
            print(f"running {file}")
            clip_loud_segments(os.path.join(config['audio'], file), config)
        except Exception as e:
            print(f"couldnt load {file} because {e}")
