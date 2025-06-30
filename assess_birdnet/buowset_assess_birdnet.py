"""Assess Birdnet performance on Buowset.

This allows for assessing how well Birdnet performs on
buowset for burrowing owl/no burrowing owl, and for
the individual call types within our labeled data.

Usage:
    python3 buowset_assess_birdnet.py /path/to/birdnet/output/
    /path/to/buowset/metadata.csv
"""
import argparse
import pandas as pd


def organize_birdnet_output(birdnet_results):
    """
    """
    birdnet_df = pd.read_pickle(birdnet_results)
    return birdnet_df

def merge_metadata(metadata, birdnet_df):
    """
    """
    meta = pd.read_csv(metadata, index_col=0)
    df_merged = meta.merge(birdnet_df, on='segment')
    df_merged = df_merged.drop(columns=['segment_duration_s', 'fold'])

    return df_merged

def assess_birdnet(merged_data):
    """
    """
    #create an x of the ground truth labels and a y of the
    #birdnet labels and just run metrics on them

def main(birdnet_results, metadata):
    """Assess birdnet.
    """
    print("Starting")
    print("Aggregating BirdNET results.")
    birdnet_df = organize_birdnet_output(birdnet_results)
    print(f"Aggregated {len(birdnet_df)} BirdNET results.")
    print(f"Matching ground truth labels to BirdNET results.")
    merged_data = merge_metadata(metadata, birdnet_df)
    print(merged_data)
    #print("Comparing BirdNET labels to ground truth.")
    #assess_birdnet(merged_data)


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
