"""

"""
import argparse
import pandas as pd
import os
import shutil

TEST_FOLD = 4
BIRD_LABEL = 0

def split_data(data, output):
    """
    """
    test = data[data['fold'] == 4]
    train = data[data['fold'] !=4]
    for index, value in test.iterrows():
        if value['label'] == 0:
            shutil.copy(value['segment_path'], (output + "/test/bird/"))
        else:
            shutil.copy(value['segment_path'], (output + "/test/noise/"))
    for index, value in train.iterrows():
        if value['label'] == 0:
            shutil.copy(value['segment_path'], (output + "/train/bird/"))
        else:
            shutil.copy(value['segment_path'], (output + "/train/noise/"))


def main(meta, output):
    """

    """
    with open(meta, 'r') as file:
        data = pd.read_csv(file)
    os.makedirs(output, exist_ok=True)
    folders = ["/test/noise/", "/test/bird/", "/train/noise/", "/train/bird/"]
    for folder in folders:
        new_path = output + folder
        os.makedirs(new_path, exist_ok=True)
    #try:
    split_data(data, output)
    #except Exception as e:
     #   print(f"Failed to split data: {e}")


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(
        description='Input Directory Path'
    )
    PARSER.add_argument('-meta', type=str,
                        help='Path to 5-fold split metadata.csv.')
    PARSER.add_argument('-output', type=str,
                        help='Path to folder with data splits.')
    ARGS = PARSER.parse_args()
    main(ARGS.meta, ARGS.output)

