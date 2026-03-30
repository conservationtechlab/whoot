"""Display RMS and Mel-Spectrogram

For a given audio file, you can visualize the RMS and
the associated Mel-Spectrogram with the same time-step to
see how they relate. Replace the filename variable with the
path to your specific audio file.

Usage:
    python3 display_rms_and_mel.py
"""
from pathlib import Path
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


FILENAME = '<path/to/audio/file.wav>'
FRAME_LENGTH = 2048
HOP_LENGTH = 512
NUM_SECONDS_OF_SLICE = 3
SAVE_PLOT = True

sound, sr = librosa.load(FILENAME, sr=None)

clip_rms = librosa.feature.rms(y=sound,
                               frame_length=FRAME_LENGTH,
                               hop_length=HOP_LENGTH)

clip_rms = clip_rms.squeeze()
peak_rms_index = clip_rms.argmax()
print(f"Peak RMS index: {peak_rms_index}")
peak_index = peak_rms_index * HOP_LENGTH + int(FRAME_LENGTH/2)
print(f"Peak index: {peak_index}")

S, phase = librosa.magphase(librosa.stft(sound))
rms = librosa.feature.rms(S=S)
fig, ax = plt.subplots(nrows=2, sharex=True)
times = librosa.times_like(rms)
ax[0].semilogy(times, rms[0], label='RMS Energy')
ax[0].set(xticks=[])
ax[0].legend()
ax[0].label_outer()
librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
                         y_axis='log', x_axis='time', ax=ax[1])
ax[1].set(title='log Power spectrogram')

if SAVE_PLOT is True:
    name = Path(FILENAME).stem
    plot_name = name + "_RMS_plot.png"
    plt.savefig(plot_name)
    print(f"Saved figure to {plot_name}")

plt.show()
