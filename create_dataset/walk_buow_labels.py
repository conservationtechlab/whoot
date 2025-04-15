"""Functions to create segments of detections of interest from wavs.


"""
from pandas as pd
import os
from pydub import AudioSegment
import logging


def setup_logger(level, filename=None):
    """
    """
    

def get_paths(home_dir):
    """
    """
    walk folder recursively and save file path of each wav in a dataframe line
    return wavs_file_paths

def create_segments(wav, filtered_labels, out_path):
    """
    """
    audio = AudioSegment.from_wav(wav)
    for _, row in filtered_labels.iterrows():

        logging.info(f"Created segment " {segment})
    return output_rows

def filter_labels_2017(wavs_file_paths, human_labels)
    """
    """
    create a sub dataframe with only the rows of the human labels that correspond to the wav
    need to parse based on burrow id first, and then "in file"
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

