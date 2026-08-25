import argparse
import random
from pydub import AudioSegment
import math
import json
import pandas as pd
from pathlib import Path
from create_segments import create_segments
from whoot import expand_window
import os


if __name__=="__main__":
    with open("/path/to/ls_labels.json", "r") as file:
        data = json.load(file)
    list_from_json = []
    out_dir = "/path/to/output/folder"
    # comma separated list of classes
    class_list  = "/path/to/class_list.txt"
    output_metadata_path = os.path.join(out_dir, "metadata.csv")
    os.makedirs(out_dir, exist_ok=True)
    for item_1 in data:
        filename = item_1['data']['audio']
        filename = Path(filename).name
        for item_2 in item_1['annotations']:
            for item_3 in item_2['result']:
                if item_3['type'] != 'labels':
                    continue
                ls_rel_start = item_3['value']['start']
                ls_rel_end = item_3['value']['end']
                for item_4 in item_3['value']['labels']:
                    label = item_4
                    list_from_json.append({
                        "birdnet_expanded_file": filename,
                        "ls_rel_start": ls_rel_start, 
                        "ls_rel_end": ls_rel_end,
                        "label": label})

    ls_data = pd.DataFrame.from_dict(list_from_json)
    with open("/path/to/expanded/birdnet/metadata.csv", "r") as meta:
        metadata = pd.read_csv(meta)
    merged_data = pd.merge(ls_data, metadata, on="birdnet_expanded_file")

    merged_data["original_start_s"] = (merged_data["offset"].astype(float) / 1000 + merged_data["ls_rel_start"].astype(float))
    merged_data["label_duration"] = (merged_data["ls_rel_end"] - merged_data["ls_rel_start"])
    # check for clips longer than 3s first, s we can use expand_window as-is
    segments_to_add = []


    for _, row in merged_data.iterrows():
        start = row["original_start_s"]
        duration = row["label_duration"]
        end = start + duration
        if duration <= 3:
            segments_to_add.append({
                "original_file_path": row["original_file_path"],
                "MANUAL ID*": row["label"],
                "OFFSET": start,
                "DURATION": duration})
        else:
            num_segments = math.ceil(duration /3)
            total_segment_duration = num_segments * 3
            extra = total_segment_duration - duration
            first_segment_start = start - (extra / 2)
            for segment_num in range(num_segments):
                segment_start = (first_segment_start + segment_num * 3)
                segments_to_add.append({
                    "original_file_path": row["original_file_path"],
                    "MANUAL ID*": row["label"],
                    "OFFSET": segment_start,
                    "DURATION": 3.0})
    segment_dataframe = pd.DataFrame(segments_to_add)
    all_output_rows = []

    for original_file, filtered_labels in segment_dataframe.groupby(
        "original_file_path"
    ):

        output_rows = create_segments(
            original_file,
            filtered_labels.copy(),
            out_dir,
            class_list,
            we=True,
            randomize=False
        )

        if output_rows is not None:
            all_output_rows.append(output_rows)

    if all_output_rows:

        final_metadata = pd.concat(
            all_output_rows,
            ignore_index=True
        )

        final_metadata.to_csv(
            output_metadata_path,
            index=False
        )

        print(final_metadata)
        print(
            f"Created {len(final_metadata)} segments "
            f"in {out_dir}"
        )

    else:
        print("No segments were created.")
