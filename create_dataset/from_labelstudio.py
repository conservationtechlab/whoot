import argparse
import random
from pydub import AudioSegment
import math
import json
import pandas as pd
from pathlib import Path


if __name__=="__main__":
    with open("/home/katie/whoot/data_exporters/label_studio_exporter/jun2026_labelstudio_export.json", "r") as file:
        data = json.load(file)
    list_from_json = []

    for item_1 in data:
        filename = item_1['data']['audio']
        filename = Path(filename).name
        for item_2 in item_1['annotations']:
            for item_3 in item_2['result']:
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
    print(ls_data)
    with open("/home/katie/whoot/data_exporters/label_studio_exporter/jun2026_results/metadata.csv", "r") as meta:
        metadata = pd.read_csv(meta)
    print(metadata)
    merged_data = pd.merge(ls_data, metadata, on="birdnet_expanded_file")

    print(merged_data)

"""    create column that does the math to figure out where label is relative to the real start, rel start + rel start 
    check for clips longer than 3s, cut them into chunks. if 3-6s, from the middle out make 2. if 6-9, make 3, one 3s in the middle, 2 on each edge.
    so one labeled segment that is 4, will become 2 segments that are 3s starting in the middle NEED TO WINDOW EXPAND EDGES ALREADY BECAUSE CANT HANDLE THAT IN CREATE_SEGMENTS 
    for original_file in merged_dataframe["original_file"]:
        audio, s = AudioSegment(original_file)
        filtered_labels = merged_dataframe.groupby(original_file)
        output_rows = create_segments(audio, filtered_labels, out_dir, class_list, we=True, randomize=False)

    if what was labeled were shortened segments of interest, include the csv that maps to original wav file
    have an outdirectory
    filter for files with labels
    for a file, if there is associated original file, use the original file, and take the relative start time from label + the rel start time from that csv to
get where the detection started in the original wav. 

   """ 
