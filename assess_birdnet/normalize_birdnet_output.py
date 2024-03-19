import pandas as pd
import numpy as np

ml_output = pd.read_csv('/home/katie/BirdNET-Analyzer/analyze_audio/analyze_audio_master.csv')

def adjust_time(row):
    chunk_number = int(row['File Name'].split('out')[1].split('.')[0])
    offset = chunk_number * 600
    row['Begin Time (s)'] += offset
    row['End Time (s)'] += offset
    return row

ml_output = ml_output.apply(adjust_time, axis=1)

total_duration = 3 * 60 * 60
all_intervals = pd.DataFrame({
    'Begin Time (s)': np.arange(0, total_duration, 3),
    'End Time (s)': np.arange(3, total_duration + 3, 3),
})

all_intervals['Label'] = 'no'

for _, row in ml_output.iterrows():
    start = row['Begin Time (s)']
    end = row['End Time (s)']
    mask = (all_intervals['Begin Time (s)'] >= start) & (all_intervals['End Time (s)'] <= end)
    all_intervals.loc[mask, 'Label'] = 'yes'

print(all_intervals.head(20))

all_intervals.to_csv('adjusted_ml_output.csv', index=False)


