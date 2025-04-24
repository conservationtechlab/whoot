"""Functions to create segments of detections of interest from wavs.


"""
import pandas as pd
import os
from pydub import AudioSegment, exceptions
import logging
from pathlib import Path
import ntpath
import uuid
import numpy as np
import random
import ntpath

def setup_logger(level, filename=None):
    """
    """
    

def get_paths(home_dir):
    """
    """
    wavs_file_paths = []
    for path, dirs, files in os.walk(home_dir):
        for file in files:
            if file.endswith('.wav'):
                new_file = os.path.join(path, file)
                wavs_file_paths.append(new_file)
    return wavs_file_paths

def create_segments(wav, filtered_labels, out_path, class_list):
    """
    """
    if filtered_labels is None:
        print(f"skipping segment creation for {wav} because it does not have labels or is not a file of interest")
        return None
    output_rows = pd.DataFrame(columns=['segment', 'label', 'segment_path', 'original_path', 'segment_duration_s', 'segment_rel_start_ms'])
    with open(class_list, 'r') as file:
        classes = file.read()
    class_list = classes.split(',')
    try:
        audio = AudioSegment.from_wav(wav)
    except exceptions.CouldntDecodeError:
        print(f"Couldn't decode: {wav}, moving to next file")
    filtered_labels['MANUAL ID*'] = filtered_labels['MANUAL ID*'].str.lower()
    df_row = 0
    path = ntpath.dirname(wav)
    for index, row in filtered_labels.iterrows():
        for call_type in class_list:
            if row['MANUAL ID*'] == call_type:
                start_time = float(row['OFFSET'])
                end_time = (start_time + float(row['DURATION']))
                start_time = start_time * 1000
                end_time = end_time * 1000
                segment = audio[start_time:end_time]
                id = uuid.uuid4()
                id = str(id) + '.wav'
                segment_path = os.path.join(out_path, id)
                segment.export(segment_path, format='wav')
                output_rows.loc[df_row] = [id, call_type, segment_path, wav, float(row['DURATION']), start_time]
                df_row += 1
        print(f"Created segment {segment_path}")
    return output_rows

# def create_birdnet_segments(wav, out_path, birdnet_class_list=None):

def create_noise_segments(wav, new_buow_rows, out_path):
    """
    Randomly select an equal number of 3s noise segments to
    the number of detections per audio file, a buffer length
    away from all of the detections in the file.
    """
    if new_buow_rows is None:
        print(f"not creating noise segments from {wav} because there were no labels or no associated labels")
        all_buow_rows = pd.DataFrame()
        return all_buow_rows
    try:
        audio = AudioSegment.from_wav(wav)
        # duration in seconds, cutting off the ms
        duration = int(len(audio) / 1000)
    except exceptions.CouldntDecodeError:
        print(f"Couldn't decode: {wav}, moving to next file")
    call_type = "no_buow"
    num = len(new_buow_rows) * 2
    seconds_array = np.zeros(duration)
    for index, row in new_buow_rows.iterrows():
        start = int((row['segment_rel_start_ms'] / 1000) - 1)
        end = int((row['segment_rel_start_ms'] / 1000) + row['segment_duration_s'])
        mask_start = max(0, start - 30)
        mask_end = min(len(seconds_array), end + 30 + 1)
        seconds_array[mask_start:mask_end] = 1
    new_sample = num / 2
    print(f"length of seconds array: {len(seconds_array)}")
    while num > new_sample:
        random_index = np.random.choice(len(seconds_array)-3)
        if seconds_array[random_index] == 0 and seconds_array[random_index + 3] == 0:
           start_time = (random_index + 1) * 1000
           end_time = (random_index + 4) * 1000
           segment = audio[start_time:end_time]
           duration_of_segment = len(segment) / 1000
           id = uuid.uuid4()
           id = str(id) + '.wav'
           segment_path = os.path.join(out_path, id)
           segment.export(segment_path, format='wav')
           new_buow_rows.loc[new_sample] = [id, call_type, segment_path, wav, duration_of_segment, start_time]
           new_sample += 1

    all_buow_rows = new_buow_rows
    return all_buow_rows

def create_csv(new_rows, output_dir):
    """
    """
    if os.path.exists(output_dir):
        pd.con
