"""Functions to create segments of detections of interest from wavs.

Functions get called in create_dataset.py.
"""
import os
import uuid
import csv
import pandas as pd
from pydub import AudioSegment, exceptions
import numpy as np
from whoot import expand_window


def get_paths(home_dir):
    """Obtain paths to every wav in the directory provided.

    Args:
        home_dir (str): Path to directory containing original wavs.

    Returns:
        list: List of all the full paths to a wav in
            the given directory.
    """
    wavs_file_paths = []
    for path, _, files in os.walk(home_dir):
        for file in files:
            if file.endswith('.wav'):
                new_file = os.path.join(path, file)
                wavs_file_paths.append(new_file)
    return wavs_file_paths


def create_segments(wav, filtered_labels, out_path, class_list, we, randomize):
    """Create the labeled segments.

    Args:
        wav (str): Path to current wav file in loop.

        filtered_labels (pd.Dataframe): The human label file reduced
            to only contain the rows of detections pertinent to the
            wav of interest.
        out_path (str): Path to directory where segment will be saved.
        class_list (str): Path to the class list that you'd like segments
            to be created for. What the manual ID's are in the human
            label file- will ignore everything that is misspelled or
            unknown labels.
        we (bool): Window expansion option.
        random (bool): Random window expansion option.

    Returns:
        pd.Dataframe: The metadata now associated with the
                      created segments for a given wav file.
    """
    print(f"creating segments for {wav}")
    if filtered_labels is None:
        print(f"skipping segment creation for {wav} because "
              "it does not have labels or is not a file of interest")
        return None
    if filtered_labels.empty:
        print("filtered labels is an empty dataframe, "
              "meaning either the sound file was not "
              "labeled or has no detections")
        return None
    output_rows = pd.DataFrame(columns=['segment',
                                        'label',
                                        'segment_path',
                                        'original_path',
                                        'segment_duration_s',
                                        'segment_rel_start_ms'])
    with open(class_list, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        classes = next(reader)
    print(classes)
    try:
        audio = AudioSegment.from_wav(wav)
    except exceptions.CouldntDecodeError:
        print(f"Couldn't decode: {wav}, moving to next file")
    filtered_labels['MANUAL ID*'] = filtered_labels['MANUAL ID*'].str.lower()
    print(filtered_labels)
    df_row = 0
    for _, row in filtered_labels.iterrows():
        for call_type in classes:
            if row['MANUAL ID*'] == call_type:
                start_time = float(row['OFFSET'])
                end_time = start_time + float(row['DURATION'])
                start_time = start_time * 1000
                end_time = end_time * 1000
                if we:
                    segment = expand_window(audio, start_time, end_time, randomize=randomize)
                else:
                    segment = audio[start_time:end_time]
                segment_id = uuid.uuid4()
                segment_id = str(segment_id) + '.wav'
                segment_path = os.path.join(out_path, segment_id)
                segment.export(segment_path, format='wav')
                output_rows.loc[df_row] = [segment_id,
                                           call_type,
                                           segment_path,
                                           wav,
                                           float(row['DURATION']),
                                           start_time]
                df_row += 1
            else:
                continue
    return output_rows


def create_noise_segments(wav, new_buow_rows, out_path):
    """Create 'no_buow' segments.
    Randomly select an equal number of 3s noise segments to
    the number of detections per audio file, a buffer length
    away from all of the detections in the file.

    Args:
        wav (str): The path to the given wav.
        new_buow_rows (pd.Dataframe): The human labeled detection
            segment metadata for the given wav.
        out_path (str): The directory where the new no_buow segments will
            go to join the human labeled segments.

    Returns:
        pd.Dataframe: The metadata for the detection as well as
            the no_buow segments created from the given wav.
    """
    if new_buow_rows is None:
        print(f"not creating noise segments from {wav} because "
              "there were no labels or no associated labels")
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
    for _, row in new_buow_rows.iterrows():
        start = int((row['segment_rel_start_ms'] / 1000) - 1)
        end = int((row['segment_rel_start_ms'] / 1000)
                  + row['segment_duration_s'])
        mask_start = max(0, start - 30)
        mask_end = min(len(seconds_array), end + 30 + 1)
        seconds_array[mask_start:mask_end] = 1
    new_sample = num / 2
    while num > new_sample:
        try:
            random_index = np.random.choice(len(seconds_array)-3)
        except ValueError:
            print(f"{wav} is not long enough to generate no_buow sounds, "
                  "keeping the detection segment but adding no no_buow")
            return new_buow_rows
        if (seconds_array[random_index] == 0 and
                seconds_array[random_index + 3] == 0):
            start_time = (random_index + 1) * 1000
            end_time = (random_index + 4) * 1000
            segment = audio[start_time:end_time]
            duration_of_segment = len(segment) / 1000
            segment_id = uuid.uuid4()
            segment_id = str(segment_id) + '.wav'
            segment_path = os.path.join(out_path, segment_id)
            segment.export(segment_path, format='wav')
            new_buow_rows.loc[new_sample] = [segment_id,
                                             call_type,
                                             segment_path,
                                             wav,
                                             duration_of_segment,
                                             start_time]
            new_sample += 1

    all_buow_rows = new_buow_rows
    return all_buow_rows
