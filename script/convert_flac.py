from pydub import AudioSegment
import glob

audio_files = glob.glob("data/burrowing_owl_dataset/audio/*")

for file in audio_files:
    song = AudioSegment.from_wav(file)
    song.export(file.replace(".wav", ".flac").replace("audio", "audio_flac"), format = "flac")