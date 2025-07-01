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
import pandas as pd
import glob
import os
import ntpath


def parse_birdnet_analysis(birdnet):
    """Create dataframe from individual birdnet result files.
    """
    bn_dict = {}
    burowl_count = 0
    result_files = glob.glob(os.path.join(birdnet, "*.txt"))
    for txt_file in result_files:
        filename = ntpath.basename(txt_file)
        filename = filename.replace("BirdNET.selection.table.txt", "wav")
        with open(txt_file, 'r') as f:
            header = f.readline().strip().split('\t')
            data = pd.read_csv(f, header=None, names=header, delimiter='\t')
        if any(data['Species Code'].str.lower() == 'burowl'):
            bn_dict[filename] = 1
            burowl_count += 1
            print(f"Found another burrowing owl, new burowl count is {burowl_count}")
        else:
            bn_dict[filename] = 0
    print("finished dict")
    birdnet_df = pd.DataFrame.from_dict(bn_dict, orient='index', columns=['bn_label'])
    birdnet_df.index.name = 'segment'
    return birdnet_df


def main(birdnet, output):
    """Save out birdnet results to a dataframe.
    """
    birdnet_df = parse_birdnet_analysis(birdnet)
    birdnet_df.to_pickle(output)
    print(birdnet_df)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Input CSV and model output'
        )
    parser.add_argument('birdnet_analysis',
                        type=str,
                        help='Path to Birdnet analysis folder.')
    parser.add_argument('output',
                        type=str,
                        help='Path to desired output for result.')
    args = parser.parse_args()
    main(args.birdnet_analysis, args.output)

