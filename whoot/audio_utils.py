"""Functions to aid in audio data pre or post processing.

"""
import random
from pydub import AudioSegment


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
        int: The relative start time in ms from the beginning of the new desired length
             segment and the actual start time of the strongly labeled audio.
    """
    clip_length = len(audio)
    duration = end_time - start_time
    # if the clip is shorter than the desired length
    if clip_length < length:
        return audio, 0
    # if the detected segment is longer than the desired length
    if duration > length:
        return audio[start_time:end_time], 0
    # if we're randomly window expanding
    if randomize:
        diff = length-duration
        offset = random.uniform(0, diff)
        new_start = start_time-offset
        # if the randomly exapanded start time is negative
        if 0 > new_start:
            return audio[0:length], start_time
        end_offset = length-(offset+duration)
        new_end = end_time + end_offset
        # if the new end is too long
        if new_end > clip_length:
            new_start = clip_length - length
            start_offset = start_time - new_start
            return audio[int(new_start):clip_length], start_offset
        return audio[int(new_start):int(new_end)], offset
    # if not random expansion, its equidistant expansion
    half_diff = (length - duration)/2
    expanded_start = start_time - half_diff
    # if expanded start is negative
    if 0 > expanded_start:
        return audio[0:length], start_time
    expanded_end = end_time + half_diff
    # if expanded end is too long
    if expanded_end > clip_length:
        new_start = clip_length - length
        start_offset = start_time - new_start
        return audio[new_start:clip_length], start_offset
    return audio[int(expanded_start):int(expanded_end)], half_diff
