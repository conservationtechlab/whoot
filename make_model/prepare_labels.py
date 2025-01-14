import os
import pandas as pd
import librosa

# Paths
csv_path = '/mnt/newnas/buow/Acoustic_Recordings/2017-2018/Results/Otay/2017/all.csv'
audio_folder = '/mnt/newnas/buow/Acoustic_Recordings/2017-2018/2017/all_audio/'
output_folder = '/mnt/newnas/buow/Acoustic_Recordings/Processed_CSVs/'

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Load master CSV file
scored_data = pd.read_csv(csv_path)

# Process each audio file
for audio_file in os.listdir(audio_folder):
    if audio_file.endswith('.wav'):
        audio_path = os.path.join(audio_folder, audio_file)

        # Get audio file duration
        try:
            y, sr = librosa.load(audio_path, sr=None)
            audio_duration = librosa.get_duration(y=y, sr=sr)
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
            continue

        # Create chunks for the file
        total_chunks = int(audio_duration // 3) + 1
        chunks_data = {
            'Chunk Start': [i * 3 for i in range(total_chunks)],
            'Chunk End': [(i + 1) * 3 for i in range(total_chunks)],
            'Label': ['no'] * total_chunks
        }
        chunks_df = pd.DataFrame(chunks_data)

        # Filter scored data for the current file (handle scattered rows)
        filtered_data = scored_data[scored_data['IN FILE'].str.strip() == audio_file]

        # Mark intervals with vocalizations
        for _, row in filtered_data.iterrows():
            # Check if TOP1MATCH is not null
            if pd.notna(row['TOP1MATCH']) and row['TOP1MATCH'] != 'null':
                try:
                    # Ensure OFFSET and DURATION are floats
                    start_time = float(row['OFFSET']) / 1000  # Convert ms to seconds
                    end_time = start_time + float(row['DURATION'])

                    # Find relevant chunks for the vocalization
                    start_chunk = int(start_time // 3)
                    end_chunk = int(end_time // 3)

                    # Mark chunks as 'yes' for the range of vocalizations
                    chunks_df.loc[start_chunk:end_chunk, 'Label'] = 'yes'
                except ValueError as ve:
                    print(f"Skipping row due to conversion error in {audio_file}: {ve}")

        # Save the labeled chunks to a new CSV file
        output_file = os.path.join(output_folder, f'{os.path.splitext(audio_file)[0]}_chunks.csv')
        chunks_df.to_csv(output_file, index=False)
        print(f"Processed {audio_file} -> {output_file}")

print("Processing complete!")
