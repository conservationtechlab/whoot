import pandas as pd
import os
import ntpath
import logging


def filter_labels_2017(wav, labels):
    """
    """
    file_name = ntpath.basename(wav)
    # isolate labels that match the wav basename
    filtered_labels = labels[labels['IN FILE'] == file_name]
    index_drop = []
    wav = str(wav)
    # ensure the labels match the site and burrow name of wav file
    for index, row in filtered_labels.iterrows():
        burrow = row['Burrow']
        bur = burrow[:-1]
        site = burrow[-1:]
        if bur not in wav:
            print(f"{bur} is not in {wav}")
            index_drop.append(index)
        if site not in wav:
            print(f"{site} is not in {wav}")
            index_drop.append(index)
    for index in index_drop:
        filtered_labels.drop(index)

    return filtered_labels

def filter_labels_2018(wavs_file_paths, human_labels):
    """
    """
    return filtered_labels
