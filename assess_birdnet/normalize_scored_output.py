import pandas as pd

scored_data = pd.read_csv('/home/katie/Downloads/ChickEmer_2017_LS27A.csv')

file_of_interest = '20170422_180000.wav'
filtered_data = scored_data[scored_data['IN FILE'] == file_of_interest]

audio_file_duration = 3 * 60 * 60

total_chunks = audio_file_duration // 3
chunks_data = {
    'Chunk Start': [i*3 for i in range(total_chunks)],
    'Chunk End': [(i+1)*3 for i in range(total_chunks)],
    'Label': ['no'] * total_chunks 
}
chunks_df = pd.DataFrame(chunks_data)

def mark_intervals(row):
    start_time = row['OFFSET']
    end_time = start_time + row['DURATION']
    start_chunk = int(start_time // 3)
    end_chunk = int(end_time // 3)
    
    if row['TOP1MATCH'] != 'null':
        chunks_df.loc[start_chunk:end_chunk, 'Label'] = 'yes'

filtered_data.apply(mark_intervals, axis=1)

print(chunks_df.head(20))

chunks_df.to_csv('formatted_data_for_20170422_180000.csv', index=False)


