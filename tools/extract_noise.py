import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

FILENAME = '/mnt/projects/PandaBear/acoustics/zoo_recordings/2025-05-23/SMM1/GPBZ_SMM01_20250523_103000.wav'  # change to path of your sound file
FRAME_LENGTH = 4096
HOP_LENGTH = 2048
NUM_SECONDS_OF_SLICE = 3

sound, sr = librosa.load(FILENAME, sr=None)
print(f"sample rate: {sr}")

clip_rms = librosa.feature.rms(y=sound,
                               frame_length=FRAME_LENGTH,
                               hop_length=HOP_LENGTH)

clip_rms = clip_rms.squeeze()
print(f"clip RMS: {clip_rms}, length of clip rms: {clip_rms.size}")
peak_rms_index = clip_rms.argmax()
print(f"Peak RMS index: {peak_rms_index}, value: {clip_rms[peak_rms_index]}")
average_rms = np.mean(clip_rms) * (3/2)

above_avg_rms = clip_rms
for index, value in enumerate(clip_rms):
    if average_rms > clip_rms[index]:
        above_avg_rms[index] = 0
    else:
        above_avg_rms[index] = 1

sum = np.sum(above_avg_rms)
print(f"num frames with above the 1.5x average rms value: {sum}")

yes_counter = 0
start_index = None
last_right_index = 0

for index, value in enumerate(above_avg_rms):
    print(f"current index in above avg rms = {index}")
    if value == 1:
        print(f"value is 1!")
        if yes_counter == 0:
            start_index = index
            print(f"newest start_index: {start_index}")
        yes_counter +=1
        print(f"yes counter : {yes_counter}")
    else:
        if yes_counter > 0:
            print(f"yes counter reached a 0 at index : {index}")
            mid_index = int((index - start_index) / 2)
            mid_index = mid_index + start_index
            real_index = mid_index * HOP_LENGTH + int(FRAME_LENGTH/2)
            half_slice_width = int(NUM_SECONDS_OF_SLICE * sr / 2)
            left_index = max(0, real_index - half_slice_width)
            print(f"left index to start clip: {left_index}")
            if left_index > last_right_index:
                right_index = real_index + half_slice_width
# current left index needs to be greater than the last right index to prevent overlap
                last_right_index = right_index

                print(f"right index to start clip: {right_index}")
                sound_slice = sound[left_index:right_index]

                sf.write(f"/home/katiegarwood/test_panda/clip{index}.wav", sound_slice, sr)
                yes_counter = 0
                print("created clip, setting yes_counter back to 0")
            else:
                print("skipping this clip because it would overlap with the last one")

if yes_counter > 0:
    stop_index = index
    mid_index = int((stop_index - start_index) / 2)
    real_index = mid_index * HOP_LENGTH + int(FRAME_LENGTH/2)
    half_slice_width = int(NUM_SECONDS_OF_SLICE * sr / 2)
    left_index = max(0, real_index - half_slice_width)
    if left_index > last_right_index:
        right_index = real_index + half_slice_width

        print(f"right index to start clip: {right_index}")
        sound_slice = sound[left_index:right_index]

        sf.write(f"/home/katiegarwood/test_panda/clip{index}.wav", sound_slice, sr)
        print("created clip, setting yes_counter back to 0")
        sf.write("/home/katiegarwood/test_panda/clip.wav", sound_slice, sr)
    else:
        print("skipping this clip because it qould overlap with the last one")

