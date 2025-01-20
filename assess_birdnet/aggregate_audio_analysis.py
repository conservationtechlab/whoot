"""Aggregate Birdnet Output Files

Script for combining the batch out output files that Birdnet (acoustic bird
species classification model) produces into one master csv including the file
name of the file that the information came from.

Args: Path to the directory containing the text files
Output: A csv with all of the birdnet results from running analysis on
multiple recordings, ignoring files that had 0 detections

python aggregate_audio_analysis.py /path/to/input/dir

"""
import os
import csv
import argparse


def main(root_dir):
    """Main function that creates a CSV and populates it with the detection
    information from Birdnet text file outputs

    Args: Path to directpry containing Birdnet results

    """
    # Scanning the root directory containing sub-direcctories
    for root, _, _ in os.walk(root_dir):
        # Get a list of all .txt files in each current directory
        txt_files = [f for f in os.listdir(root) if f.endswith('.txt')]

        if txt_files:
            # Sorting.txt files based on file names - int sorting time
            sorted_txt_files = sorted(txt_files)
            subdirectory_name = os.path.basename(root)

            # Create the CSV file for the current folder
            output_csv_file = os.path.join(root,
                                           f"{subdirectory_name}_master.csv")
            csv_headers = [
                "Selection",
                "View",
                "Channel",
                "Begin Time (s)",
                "End Time (s)",
                "Low Freq (Hz)",
                "High Freq (Hz)", "Species Code",
                "Common Name",
                "Confidence",
                "File Name"
            ]

            with open(
                output_csv_file, mode='w', newline='', encoding='utf8'
            ) as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(csv_headers)

                for txt_file in sorted_txt_files:
                    with open(
                        os.path.join(root, txt_file), 'r', encoding='utf8'
                    ) as txtfile:
                        lines = txtfile.readlines()[1:]
                        for line in lines:
                            data = line.strip().split('\t')
                            file_name = os.path.splitext(txt_file)[0]
                            data.append(file_name)
                            csv_writer.writerow(data)

            print(f"File for directory '{root}' created successfully.")


if __name__ == '__main__':
    # Create an argument parser
    parser = argparse.ArgumentParser(
        description='Input Directory Path'
        )

    # Create and parse arguements
    parser.add_argument('root_dir', type=str, help='Directory path to files')
    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the main function
    main(args.root_dir)
