"""Expand birdnet detections into longer clips for validation.

This script takes in t combined table of birdnet detections for a given
group of wav files. It checks for overlap, and then creates an expanded clip
in order to provide acoustic context while validating the birdnet detections.
It also outputs a pkl file containing a dictionary of audio info including
the offset and duration of the label, as well as the label for each clip.

"""
import argparse
from whoot import check_overlap_dict
import pandas as pd
import pickle


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(
        description='Input config path'
    )
    PARSER.add_argument('-results', type=str,
                        help='Path to Birdnet concatenated results file.')
    PARSER.add_argument('-output', type=str,
                        help='Path to output dir for metadata and segments.')
    ARGS = PARSER.parse_args()
    results = pd.read_csv(ARGS.results)
    out_dir = ARGS.output

    results['File'] = results['File'].str.replace('Volumes/BUOW', 'mnt/restorage')

    all_data = {
        "audio": [],
        "labels": [],
    }

    metadata = []

    # creates shortened segments that combine overlaps into 1, provides results in dataframe
    for file_path, detections in results.groupby("File"):
        metadata_dict, metadata_list = check_overlap_dict(file_path, detections, out_dir)

        all_data["audio"].extend(metadata_dict["audio"])
        all_data["labels"].extend(metadata_dict["labels"])

        metadata.extend(metadata_list)

    dataframe = pd.DataFrame(metadata)

    with open("output.pkl", "wb") as file:
        pickle.dump(all_data, file)

    dataframe.to_csv("metadata.csv", index=False)
