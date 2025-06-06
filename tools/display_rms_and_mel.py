import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

FILENAME = '/mnt/projects/PandaBear/acoustics/zoo_recordings/2025-05-23/SMM1/GPBZ_SMM01_20250523_103000.wav'  # change to path of your sound file
FRAME_LENGTH = 2048
HOP_LENGTH = 512
NUM_SECONDS_OF_SLICE = 3

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


plt.show()
