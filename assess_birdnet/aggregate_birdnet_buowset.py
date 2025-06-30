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


def parse_birdnet_analysis(birdnet):
    """Create dataframe from individual birdnet result files.
    """
    open each txt file in directory and obtain the label for
    the segment, and associate it with a key value where the
    key is the filename minus the birdnet stuff and then the
    label is the value
    convert that whole thing to a df.
    return birdnet_df


def main(birdnet, output):
    """Save out birdnet results to a dataframe.
    """
    birdnet_df = parse_birdnet_analysis(birdnet)
    save birdnet_df as "output".pkl

if __name__ = '__main__':
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

