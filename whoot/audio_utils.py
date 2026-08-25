"""Functions to aid in audio data pre or post processing.

"""
import random
from pydub import AudioSegment
from pathlib import Path
import pandas as pd


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
        int: The relative start time in ms from the beginning of the new
             desired length segment and the actual start time of the strongly
             labeled audio.
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


def check_overlap_dict(file_path, detections, output_dir):
    """Check for overlap with other detections before expanding window
       and create a dictionary with the audio, the new path, the duration and 
       offset of the detection within the newly expanded window.

    Args:
        file_path:
        detections:
        output_dir:

    Returns:
        dict: A dictionary containing the clip path, offset/duration and label.
    """

    audio = AudioSegment.from_wav(file_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    detections = detections.sort_values("Start (s)")

    groups = []

    for _, row in detections.iterrows():
        start = int(row["Start (s)"] * 1000)
        end = int(row["End (s)"] * 1000)

        detection = {
            "start": start,
            "end": end,
            "label": row["Common name"],
        }


        if groups and start - 3500 <= groups[-1]["end"] + 3500:
            groups[-1]["end"] = max(groups[-1]["end"], end)
            groups[-1]["detections"].append(detection)
        else:
            groups.append({
                "start": start,
                "end": end,
                "detections": [detection],
            })

    metadata_dict = {
        "audio": [],
        "labels": [],
    }

    dataframe_list = []

    for i, group in enumerate(groups):
        group_start = group["start"]
        group_end = group["end"]
        length = (group_end - group_start) + 7000

        clip, group_offset = expand_window(
            audio,
            group_start,
            group_end,
            length,
            randomize=False,
        )

        segment_name = f"{Path(file_path).stem}_{i}.wav"
        output_path = output_dir / segment_name
        clip.export(output_path, format="wav")
        dataframe_dict = {
            "ls_filename": str(segment_name),
            "original_file_path": str(file_path),
            "offset": group_start,
            "duration": length
        }
        dataframe_list.append(dataframe_dict)

        for detection in group["detections"]:
            detection_offset = (
                group_offset
                + detection["start"]
                - group_start
            )

            metadata_dict["audio"].append({
                "bytes": None,
                "path": str(output_path),
                "offset": detection_offset / 1000,
                "duration": (
                    detection["end"] - detection["start"]
                ) / 1000,
            })

            metadata_dict["labels"].append(detection["label"])

    return metadata_dict, dataframe_list
