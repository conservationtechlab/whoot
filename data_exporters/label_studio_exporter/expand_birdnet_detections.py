"""Expand birdnet detections into longer clips for validation.

This script takes in the combined table of birdnet detections for a given
group of wav files. It checks for overlap, and then creates an expanded clip
in order to provide acoustic context while validating the birdnet detections.
It also outputs a pkl file containing a dictionary of audio info including
the offset and duration of the label, as well as the label for each clip.
This can be directly used to upload the birdnet start/stop labels in
labelstudio. The metadata.csv can be used to map the shortened segments
back to the longer original files to be able to create window
expanded segments from strongly labeled data once it has been labeled
in labelstudio or elsewhere.

Usage:
    python3 expand_birdnet_detections.py
     -results /path/to/birdnet/combined/results.csv
     -output /path/to/desired/output/folder/

"""
import argparse
import pickle
from whoot import check_overlap_dict
import pandas as pd


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

    results['File'] = results['File'].str.replace('Volumes/BUOW',
                                                  'mnt/restorage')

    all_data = {
        "audio": [],
        "labels": [],
    }

    metadata = []

    # creates shortened segments that combine overlaps into 1
    for file_path, detections in results.groupby("File"):
        metadata_dict, metadata_list = check_overlap_dict(file_path,
                                                          detections,
                                                          out_dir)

        all_data["audio"].extend(metadata_dict["audio"])
        all_data["labels"].extend(metadata_dict["labels"])

        metadata.extend(metadata_list)

    dataframe = pd.DataFrame(metadata)

    with open(f"{out_dir}/output.pkl", "wb") as file:
        pickle.dump(all_data, file)

    dataframe.to_csv(f"{out_dir}/metadata.csv", index=False)
