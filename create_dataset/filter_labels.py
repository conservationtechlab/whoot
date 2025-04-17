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

    filtered_labels.drop(index_drop)

    return filtered_labels

def filter_labels_2018(wav, labels):
    """
    Because we do not have full file paths, we need to ensure that there
    are not duplicate .wav file names that are associated with different burrows/sites.
    If we just use the all label file, it would be difficult to determine which burrow/site
    is correct for the wav file, because the file paths are inconsistent. This function
    chooses the label file to use based on the wav name, and then obtains the labels for
    that site/burrow within that folder so that there's no question that it's for that
    site/burrow. 2017 is formatted very differently and we are able to back out the burrow/site
    from the path to the wav and other information in the all labels file. 
    """
    file_name = ntpath.basename(wav)
    path_name = ntpath.dirname(wav)
    basepath = os.path.basename(path_name)
    if basepath == "ClassificationResults" or basepath == "Classification_Results":
        print(f"skipping {wav} because it's basepath is {basepath}")
        # skipping extra wav files that exist as duplicates of our wavs of interest within these sub dirs
        return None
    # some of the folders have an underscore and some do not
    path_labels = []
    path_labels.append(path_name + "/ClassificationResults/")
    path_labels.append(path_name + "/Classification_Results/")
    path_to_results = None
    # checking if it's the one with an underscore vs not
    for path in path_labels:
        exists = os.path.exists(path)
        if exists == True:
            path_to_results = path
        else:
            print(f"{path} does not exist")
            continue
    if path_to_results == None:
        # skipping wav files that are an exception to this folder structure because they're
        # not the wav files of interest
        print(f"skipping {wav} because it's not a file of interest")
        return None
    filtered_labels = labels[labels['IN FILE'] == file_name]
    index_to_drop = []
    # iterating the columns in labels that match the wav file name
    for index, row in filtered_labels.iterrows():
        check_path = os.path.join(path_to_results, row['Fled_2018_LS133_SM1.csv '].strip())
        # there's a column in the all labels file that has the file name of the subset label file that
        # the all labels file was aggregated from, and if the wav file path leads us to
        # the label file listed in the all labels file, then it will be apart of the filtered
        # labels for that wav. this needs to be checked in case 2 wav files have the same
        # file name, but are from different burrows/sites. 
        # it's worth noting that this could be done a different way, using the subset label files
        # for each burrow/site labels, but you'd still need the all labels file to validate, so it
        # just felt like more steps
        if os.path.isfile(check_path):
            continue
        else:
            index_to_drop.append(index)
    # if there were labels associated with a different wav file that happened to have the same
    # name, this will drop the labels associated with a different burrow/site
    filtered_labels = filtered_labels.drop(index_to_drop)

    return filtered_labels


# TODO: There's one subset label file that has no column names, so that error needs to be dealt with. It will currently ignore that one
