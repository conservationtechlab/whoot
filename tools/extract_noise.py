"""Extract noisy segments from a wav file.

Takes in a wav file and an outpath to store
the 3 second segments that contain an RMS value above
the average RMS for that wav file.
"""
import uuid
import librosa
import numpy as np
import soundfile as sf
import audioread


def clip_loud_segments(file, config, rows):  # pylint: disable=too-many-locals
    """Extract loud segments from a wav file.

    If a section of audio RMS is 1.5x above the average
    RMS of the whole file, that section will be stored as
    its own segment without overlapping.

    Args:
        file (str): The path of the current wav file.
        config (dict): The config values.
        rows (list): List where metadata will be continuously
            added before it is saved out as a file.

    Returns:
        int: Number of clips generated
        None: Only if audio file was unreadable to exit loop.

    Raises:
        audioread.exceptions.NoBackendError: If audio file is
            not readable.
    """
    index = None
    filename = file
    frame_length = config['frame_length']
    hop_length = config['hop_length']
    num_sec_slice = config['num_sec_slice']
    try:
        sound, sr = librosa.load(filename, sr=None)
    except audioread.exceptions.NoBackendError:
        print(f"skipping {file}, corrupt?")
        return None
    print(f"sample rate: {sr}")

    above_avg_rms = find_peaks(frame_length, hop_length, sound)

    yes_counter = 0
    start_index = None
    last_right_index = 0
    clips_saved = 0
    for index, value in enumerate(above_avg_rms):
        if value == 1:
            if yes_counter == 0:
                start_index = index
            yes_counter += 1
        else:
            if yes_counter > 0:
                mid_index = int((index - start_index) / 2)
                mid_index = mid_index + start_index
                real_index = mid_index * hop_length + int(frame_length/2)
                half_slice_width = int(num_sec_slice * sr / 2)
                left_index = max(0, real_index - half_slice_width)
                if left_index > last_right_index:
                    right_index = real_index + half_slice_width
                    # left index needs to be greater than the last right
                    last_right_index = right_index + 1
                    name = store(sound, left_index, right_index, config, sr)
                    write_row(file, name, left_index, rows)
                    yes_counter = 0
                    print(f"created {name}, setting yes_counter back to 0")
                    clips_saved += 1

    if yes_counter > 0:
        right_index = index
        mid_index = int((right_index - start_index) / 2)
        real_index = mid_index * hop_length + int(frame_length/2)
        half_slice_width = int(num_sec_slice * sr / 2)
        left_index = max(0, real_index - half_slice_width)
        if left_index > last_right_index:
            name = store(sound, left_index, right_index, config, sr)
            write_row(file, name, left_index, rows)
            clips_saved += 1

    return clips_saved


def store(sound, left_index, right_index, config, sr):
    """Store the clipped segment.

    Args:
        sound (numpy.ndarray): The original wav file.
        left_index (int): Relative start time of the clip in ms.
        right_index (int): Relative end time of the clip in ms.
        config (dict): Config values.
        sr (int): The sample rate of the audio.

    Returns:
        str: The name of the new segment.
    """
    segment_id = uuid.uuid4()
    name = config['out'] + str(segment_id) + ".wav"
    sound_slice = sound[left_index:right_index]
    sf.write(name, sound_slice, sr)

    return name


def write_row(file, name, left_index, rows):
    """Write metadata to a row.

    Args:
        file (str): Original file where clip came from.
        name (str): Name of clip created.
        left_index (int): Relative start time of segment in ms.
        rows (list): List of metadata.
    """
    rows.append({
        "original_path": file,
        "clip_path": name,
        "rel_start_ms": left_index,
    })


def find_peaks(frame_length, hop_length, sound):
    """Find peak RMS moments in a sound file.

    Args:
        frame_length (int): Window size.
        hop_length (int): Overlap between frames.
        sound (numpy.ndarray): The audio as a time series array.

    Returns:
        numpy.ndarray: The array containing each frame as an index
                       with values corresponding to whether that
                       frame exceeded the avg RMS or not.
    """
    clip_rms = librosa.feature.rms(y=sound,
                                   frame_length=frame_length,
                                   hop_length=hop_length)

    clip_rms = clip_rms.squeeze()
    average_rms = np.mean(clip_rms) * (3/2)
    above_avg_rms = clip_rms

    for index, _ in enumerate(clip_rms):
        if average_rms > clip_rms[index]:
            above_avg_rms[index] = 0
        else:
            above_avg_rms[index] = 1

    num_frames = np.sum(above_avg_rms)
    print(f"num frames with above the 1.5x average rms value: {num_frames}")

    return above_avg_rms
