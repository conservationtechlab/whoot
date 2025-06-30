"""Assess Birdnet performance on Buowset.

This allows for assessing how well Birdnet performs on
buowset for burrowing owl/no burrowing owl, and for
the individual call types within our labeled data.

Usage:
    python3 buowset_assess_birdnet.py /path/to/birdnet/output/
    /path/to/buowset/metadata.csv
"""


def organize_birdnet_output(birdnet_results):
    """
    """
    open up the pkl as dataframe
    return birdnet_df

def merge_metadata(metadata, birdnet_df):
    """
    """
    open metadata as a df and merge on the filename
    with column of the real_label and a column 
    forthe birdnet label
    return merged_data

def assess_birdnet(merged_data):
    """
    """
    create an x of the ground truth labels and a y of the
    birdnet labels and just run metrics on them

def main(birdnet_results, metadata):
    """Assess birdnet.
    """
    print("Starting")
    print*"Aggregating BirdNET results.")
    birdnet_df = organize_birdnet_output(birdnet_results)
    print(f"Aggregated {len(birdnet_df)} BirdNET results.")
    print(f"Matching ground truth labels to BirdNET results.")
    merged_data = merge_metadata(metadata, birdnet_df)
    print("Comparing BirdNET labels to ground truth.")
    assess_birdnet(merged_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Input Directory Path'
        )
    parser.add_argument('birdnet_results',
                        type=str,
                        help='Path to Birdnet results for padded buowset.')
    parser.add_argument('metadata',
                        type=str,
                        help='Path to buowset metadata file.')
    args = parser.parse_args()
    main(args.birdnet_results, args.metadata)
