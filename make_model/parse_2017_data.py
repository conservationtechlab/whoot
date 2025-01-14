import os
import pandas as pd
import librosa

csv_path = '/mnt/newnas/buow/Acoustic_Recordings/2017-2018/Results/Otay/2017/all.csv'
audio_folder = '/mnt/newnas/buow/Acoustic_Recordings/2017-2018/2017/all_audio/'
output_folder = '/mnt/newnas/buow/Acoustic_Recordings/2017-2018/2017/all_audio_for_training/'

os.makedirs(output_folder, exist_ok=True)

scored_data = pd.read_csv(csv_path)

for audio_file in os.listdir(audio_folder):
    if audio_file.endswith('.wav'):
        audio_path = os.path.join(audio_folder, audio_file)

        try:
            y, sr = librosa.load(audio_path, sr=None)
            audio_duration = librosa.get_duration(y=y, sr=sr)
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
            continue

        total_chunks = int(audio_duration // 3) + 1
        chunks_data = {
            'Chunk Start': [i * 3 for i in range(total_chunks)],
            'Chunk End': [(i + 1) * 3 for i in range(total_chunks)],
            'Label': ['no'] * total_chunks
        }
        chunks_df = pd.DataFrame(chunks_data)

        filtered_data = scored_data[scored_data['IN FILE'] == audio_file]

        for _, row in filtered_data.iterrows():
            if row['TOP1MATCH'] != 'null':
                start_time = float(row['OFFSET']) / 1000
                end_time = start_time + float(row['DURATION'])

                for i in range(len(chunks_df)):
                    chunk_start = chunks_df.loc[i, 'Chunk Start']
                    chunk_end = chunks_df.loc[i, 'Chunk End']
                    if (start_time < chunk_end and end_time > chunk_start):
                        chunks_df.loc[i, 'Label'] = 'yes'

        output_file = os.path.join(output_folder, f'{os.path.splitext(audio_file)[0]}_chunks.csv')
        chunks_df.to_csv(output_file, index=False)
        print(f"Processed {audio_file} -> {output_file}")

print("Processing complete!")
