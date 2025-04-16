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

def create_noise_segments(wav, filtered_labels, num, out_path):
    """
    """


def create_csv(new_rows):
    """
    """

