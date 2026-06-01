"""Correlating the wav paths with the labels for 2017 and 2018.

The label file format is different for the 2018 and 2017 label
files. This means we use different information in those files to
ensure the wav file we found in the folder corresponds to the
label in the label file. Depending on the label file, one
of these two functions gets called to ensure we're dealing
with the proper wav file and only the labels that correspond
to that wav file.
"""
import os
import ntpath
from pathlib import Path


def custom_filter(wav, labels):
    """
    """
    print("Custom filter is not defined, please edit filter_labels.py")


def default_filter(wav, labels):
    """We have a subset of data we'd like to
    create a dataset out of, there are no duplicates
    but we need to filer the label to the corresponding
    wav file.

    Args:
        wav (str): The current wav file.
        labels (pd.DataFrame): All of the labels.

    Returns:
        pd.DataFrame: The labels associated with the wav of interest.

    """
    file_name = ntpath.basename(wav)
    labels['match_file'] = labels['path'].apply(lambda x: Path(x).name)
    filtered_labels = labels[labels['match_file'] == file_name]
    return filtered_labels


def filter_labels_2017(wav, labels):
    """Filter labels from 2017 data.

    Args:
        wav (str): The current wav file.
        labels (pd.DataFrame): All of the labels.

    Returns:
        pd.DataFrame: The labels associated with the wav of interest.
    """
    file_name = ntpath.basename(wav)
    # isolate labels that match the wav basename
    filtered_labels = labels[labels['IN FILE'] == file_name]
    index_drop = []
    wav = str(wav)
    # ensure the labels match the site and burrow name of wav file
    # this step is crucial, it catches accidential duplicates of wav files
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

    filtered_labels = filtered_labels.drop(index_drop)
    return filtered_labels


def filter_labels_2018(wav, labels):
    """Filter labels from 2018 data.

    Args:
        wav (str): The current wav file.
        labels (pd.DataFrame): All of the labels.

    Returns:
        pd.DataFrame: The labels associated with the wav of interest.
    """
    file_name = ntpath.basename(wav)
    path_name = ntpath.dirname(wav)
    basepath = os.path.basename(path_name)
    if basepath in ('ClassificationResults', 'Classification_Results'):
        print(f"skipping {wav} because it's basepath is {basepath}")
        # skipping extra wav files that exist as duplicates in these sub dirs
        return None
    # some of the folders have an underscore and some do not
    path_labels = []
    path_labels.append(path_name + "/ClassificationResults/")
    path_labels.append(path_name + "/Classification_Results/")
    path_to_results = None
    # checking if it's the one with an underscore vs not
    for path in path_labels:
        exists = os.path.exists(path)
        if exists is True:
            path_to_results = path
        else:
            print(f"{path} does not exist")
            continue
    if path_to_results is None:
        print(f"skipping {wav} because it's not a file of interest")
        return None
    filtered_labels = labels[labels['IN FILE'] == file_name]
    index_to_drop = []
    # iterating the columns in labels that match the wav file name
    for index, row in filtered_labels.iterrows():
        stripped = row['Fled_2018_LS133_SM1.csv '].strip()
        check_path = os.path.join(path_to_results, stripped)
        if os.path.isfile(check_path):
            continue
        if stripped == 'EarBreed_2018_LS128_SM10A.csv':
            check_path = os.path.join(path_to_results,
                                      'EarBreed_LS128_SM10A.csv')
            if not os.path.isfile(check_path):
                index_to_drop.append(index)
        else:
            index_to_drop.append(index)
    filtered_labels = filtered_labels.drop(index_to_drop)
    return filtered_labels
