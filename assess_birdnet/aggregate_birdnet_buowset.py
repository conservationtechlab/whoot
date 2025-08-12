"""Create a dataframe out of birdnet results.

When you run BirdNET analyze on wav files, it outputs a
result txt file per each wav. We need to aggregate all
of the results into 1 dataframe and saved out so we can
reference it later when we analyze the birdnet results
for buowset.

Usage:
    python3 aggregate_birdnet_buowset.py /path/to/birdnet/
    analyzer/folder/ /path/to/output.pkl
"""
import argparse
import glob
import os
import ntpath
import pandas as pd


def parse_birdnet_analysis(birdnet):
    """Create dataframe from individual birdnet result files.

    Args:
        birdnet (str): Path to the birdnet results.

    Returns:
        pandas.DataFrame: Birdnet results as a single dataframe.
    """
    bn_dict = {}
    burowl_count = 0
    result_files = glob.glob(os.path.join(birdnet, "*.txt"))
    for txt_file in result_files:
        filename = ntpath.basename(txt_file)
        filename = filename.replace("BirdNET.selection.table.txt", "wav")
        with open(txt_file, 'r') as file:
            header = file.readline().strip().split('\t')
            data = pd.read_csv(file, header=None, names=header, delimiter='\t')
        if any(data['Species Code'].str.lower() == 'burowl'):
            bn_dict[filename] = 1
            burowl_count += 1
            print(f"New burowl count is {burowl_count}")
        else:
            bn_dict[filename] = 0
    print("finished dict")
    birdnet_df = pd.DataFrame.from_dict(bn_dict,
                                        orient='index',
                                        columns=['bn_label'])
    birdnet_df.index.name = 'segment'
    return birdnet_df


def main(birdnet, output):
    """Save out birdnet results to a dataframe.

    Args:
        birdnet (str): Path to the birdnet results files.
        output (str): Filename for the output pkl.
    """
    birdnet_df = parse_birdnet_analysis(birdnet)
    birdnet_df.to_pickle(output)
    print(birdnet_df)


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(
        description='Input CSV and model output'
        )
    PARSER.add_argument('birdnet_analysis',
                        type=str,
                        help='Path to Birdnet analysis folder.')
    PARSER.add_argument('output',
                        type=str,
                        help='Path to desired output for result.')
    ARGS = PARSER.parse_args()
    main(ARGS.birdnet_analysis, ARGS.output)
