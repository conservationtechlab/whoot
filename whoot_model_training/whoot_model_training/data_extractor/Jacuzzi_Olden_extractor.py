"""Ceates Dataset from the Xeno-Canto Data Downlaoder tool.

See data_downloader/xc.py
"""

# import numpy as np
# from ..dataset import AudioDataset

# import json
# import pandas as pd


# def one_hot_encode(row: dict, classes: list):
#     """One hot Encodes a list of labels.

#     Args:
#         row (dict): row of data in a dataset containing a labels column
#         classes: a list of classes
#     """
#     one_hot = np.zeroes(len(classes))
#     one_hot[row["labels"]] = 1
#     row["labels"] = np.array(one_hot, dtype=float)
#     return row

# def Jacuzzi_Olden_Extractor(root_path):
#     audio_path = f"{root_path}/training/audio"
#     train_df = pd.read_csv(
#         f"{root_path}/training/training_data_annotations.csv"
#     )
#     train_df["labels"] = train_df["labels"].str.split(",")
#     train_df["file_path"] = train_df["audio_subdir"].apply(
#         lambda folder: f"{audio_path}/{folder}/"
#     ) + train_df["file"].apply(lambda path: path + ".wav")

#     test_df = pd.read_csv(f"{root_path}/test/test_data_annotations.csv")
#     test_df["labels"] = test_df["labels"].str.split(",")
#     test_df["file"] = test_df["file"].str.findall(
#         r"-0.\d+_([\w.]+).wav").apply(lambda x: x[0])
#     test_df["file_path"] = test_df["focal_class"].apply(
#         lambda folder: f"{audio_path}/{folder}/"
#     ) + test_df["file"].apply(lambda path: path + ".wav")

#     return train_df, test_df

#     # TODO
#     # Convert to AudioDataset
#     # Convert Labels to right format
#     # Convert audio type
#     # Done
