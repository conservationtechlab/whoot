Tools for handling unlabeled raw audio.

To investigate and understand your raw audio data better, 
and to be able to isolate potentially significant acoustic
events to reduce time labeling.

run_extract_noise.py will generate 3s clips from larger wav files
where the RMS of that segment exceeded the average RMS of the 
entire clip. This can highlight loud events in an audio file.

display_rms_and_mel.py will give a visual graph with the mel
spectrogram and RMS chart for a given wav for a sanity check
and to get a better idea of what the spectrogram looks like for
a given RMS peak.
