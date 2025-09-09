"""Functions to aid in audio data pre or post processing.

"""
import random
from pydub import AudioSegment
from pydub.exceptions import TooManyMissingFrames


def expand_window(audio, start_time, end_time, length=3000, randomize=False):
    """Expand the window size of a detection.

    If you have audio segments with a designated start and stop time
    for a detection, you can choose to expand the window to generate
    an expanded detection window audio clip, clips longer than the desired
    length will neither be expanded or shortened.

    Args:
        audio (AudioSegment.from_wav): An audio segment object.
        start_time (float): Millisecond in wav file where the detection began.
        end_time (float): Millisecond in wav file where detection ended.
        length (int): Duration in millisecond of the desired window length.
        randomize (bool): If the window expansion will have the detection
            centered or have randomized beginning and end window expansion.

    Returns:
        AudioSegment: The audio sample with expanded window of the detection.
    """
    clip_length = len(audio)
    duration = end_time - start_time
    if clip_length < length:
        return audio
    if duration > length:
        segment = audio[start_time:end_time]
        return segment
    if randomize:
        diff = length-duration
        offset = random.uniform(0, diff)
        new_start = start_time-offset
        end_offset = length-(offset+duration)
        new_end = end_time + end_offset
        if new_end > clip_length:
            end_diff = clip_length - start_time
            new_start = start_time - (length - end_diff)
            segment = audio[new_start:clip_length]
        else:
            segment = audio[new_start:new_end]
    else:
        half_diff = (length - duration)/2
        expanded_start = start_time - half_diff
        expanded_end = end_time + half_diff
        if expanded_end > clip_length:
            end_diff = clip_length - start_time
            new_start = start_time - (length - end_diff)
            segment = audio[new_start:clip_length]
        else:
            segment = audio[expanded_start:expanded_end]

    return segment
