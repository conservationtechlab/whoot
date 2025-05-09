from pydub import AudioSegment
import os


path = "/home/katiegarwood/Downloads/audio_copy_birdnet/audio/"
output = "/home/katiegarwood/Downloads/audio_copy_birdnet_padded/"
for file in os.listdir(path):
    filepath = os.path.join(path, file)
    pad_ms = 3000  # Add here the fix length you want (in milliseconds)
    audio = AudioSegment.from_wav(filepath)
    if len(audio) < pad_ms:
        silence = AudioSegment.silent(duration=pad_ms-len(audio)+1)
        padded = audio + silence  # Adding silence after the audio
        full_path = os.path.join(output, file)
        padded.export(full_path, format='wav')
    else:
        full_path = os.path.join(output, file)
        audio.export(full_path, format='wav')
