import pandas as pd

def make_x_and_y(metadata, embeds):
    x_train = [] # 4 of the folds
    y_train = [] # 4 of the folds
    x_test = [] # the small one 20%, 1 folds worth
    y_test = [] # the small one 20%, 1 folds worth
    for row in metadata:
        filename = filename strip .wav
        filename = filename+.birdnet.embeddings.txt
        embedpath = embeds + filename
        dfb = pd.read_csv(embedpath,
                          delimiter="[,\t]",
                          engine='python',
                          header=None)
        dfb_stripped = dfb.drop(dfb.columns[:2], axis=1)
        if row['fold'] == range(0, 3):
            x_train.append(dfb)
            if row['label'] == range(0, 4):
                y_train.append(1)
            else:
                y_train.append(0)
        else:
            x_test.append(dfb)
            if row['label'] == range(0, 4):
                y_test.append(1)
            else:
                y_test.append(0)

def make_svm():
    """
    """
    x_train, y_train, x_test, y_test = make_x_and_y(stuff, stuff) 


if __name__="__main__":
    arg parser for path to audio and birdnet embeddings for padded samples
    path to metadata with fold info and label
    make_svm()

