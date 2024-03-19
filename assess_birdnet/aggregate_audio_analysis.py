import os
import csv
import argparse

def main(root_dir):
    # Scanning the root directory containing sub-direcctories
    for root, _, _ in os.walk(root_dir):
        # Get a list of all .txt files in each current directory
        txt_files = [f for f in os.listdir(root) if f.endswith('.txt')]

        if txt_files:
            # Sorting.txt files based on file names - int sorting time
            sorted_txt_files = sorted(txt_files)
            subdirectory_name = os.path.basename(root)

            # Create the CSV file for the current folder
            output_csv_file = os.path.join(root, f"{subdirectory_name}_master.csv")
            csv_headers = [
                "Selection", "View", "Channel", "Begin Time (s)", "End Time (s)", 
                "Low Freq (Hz)", "High Freq (Hz)", "Species Code", "Common Name", "Confidence", "File Name"
            ]

            with open(output_csv_file, mode='w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(csv_headers)

                for txt_file in sorted_txt_files:
                    with open(os.path.join(root, txt_file), 'r') as txtfile:
                        lines = txtfile.readlines()[1:]  # Ignoring the *individual headers
                        for line in lines:
                            data = line.strip().split('\t')
                            file_name = os.path.splitext(txt_file)[0]
                            data.append(file_name)  # Adding the file name to the data
                            csv_writer.writerow(data)

            print(f"Master CSV file for directory '{root}' created successfully.")

if __name__ == '__main__':
    # Create an argument parser
    parser = argparse.ArgumentParser(
        description='Input and Output Directory Paths  for Audio Files Aggregation'
        )

    # Create and parse arguements
    parser.add_argument('root_dir', type=str, help='Path to Root Directory containing sub directories with audio files and corresponding txt files')
    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the main function
    main(args.root_dir)
