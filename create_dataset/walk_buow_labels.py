"""Functions to create segments of detections of interest from wavs.


"""
import pandas as pd
import os
from pydub import AudioSegment
import logging
from pathlib import Path
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

def create_segments(wav, filtered_labels, out_path):
    """
    """
    audio = AudioSegment.from_wav(wav)
    for _, row in filtered_labels.iterrows():

        logging.info(f"Created segment {segment}")
    return output_rows

def filter_labels_2017(wav, labels):
    """
    """
    file_name = ntpath.basename(wav)
    # isolate labels that match the wav basename
    filtered_labels = labels[labels['IN FILE'] == file_name]
    index_drop = []
    wav = str(wav)
    # ensure the labels match the site and burrow name of wav file
    for index, row in filtered_labels.iterrows():
        burrow = row['Burrow']
        bur = burrow[:-1]
        site = burrow[-1:]
        if bur not in wav:
            print(f"{bur} is not in {wav}")
            index_drop.append(index)
        if site not in wav:
            print(f"{site} is not in {wav}")
            index_drop.append(index)
    for index in index_drop:
        filtered_labels.drop(index)

    return filtered_labels

def filter_labels_2018(wavs_file_paths, human_labels):
    """
    """
    return filtered_labels

def create_noise_segments(wav, filtered_labels, num, out_path):
    """
    """


def create_csv(new_rows):
    """
    """

