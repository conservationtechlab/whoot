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
from pathlib import Path
import os
import yaml
import pandas as pd
from extract_noise import clip_loud_segments


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(
        description='Path to config file.'
    )
    PARSER.add_argument('-config', type=str,
                        help='Path to config.')
    ARGS = PARSER.parse_args()
    with open(ARGS.config, 'r', encoding='UTF-8') as f:
        config = yaml.safe_load(f)
    if not os.path.exists(config['out']):
        print(f"{config['out']} does not exist, creating directory.")
        os.makedirs(config['out'], exist_ok=True)
    all_wav_files = [str(p) for p in Path(config['audio']).rglob("*.wav")]
    rows = []
    for wav in all_wav_files:
        print(f"running {wav}")
        saved = clip_loud_segments(wav, config, rows)
        if saved is not None:
            print(f"Saved {saved} clips from {wav}")

    metadata = pd.DataFrame(rows)
    meta_name = os.path.join(config['out'], "metadata.csv")
    metadata.to_csv(meta_name, index=False)
    print(f"Saved metadata to {meta_name}")
