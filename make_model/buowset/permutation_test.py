"""
"""
import pandas as pd
import argparse
import pickle
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import os
import numpy as np
import csv
from make_svm import make_x_and_y

def permutation_test(meta, embeds):
    """
    """
    data = pd.read_csv(meta, index_col=0)
    permutated_accuracies = []
    permutation_iters = 100
    x_train, y_train, x_test, y_test = make_x_and_y(data, embeds)
    for i in range(permutation_iters):
        np.random.shuffle(y_train)
        np.random.shuffle(y_test)
        svm = SVC(class_weight='balanced', probability=True)
        svm.fit(x_train, y_train)
        y_pred_default = svm.predict(x_test)
        permutated_accuracies.append(metrics.accuracy_score(y_pred_default, y_test)*100)
    print(f"Average permutated accuracy is: {np.mean(permutated_accuracies)}")

def main(meta, embeds):
    """
    """
    permutation_test(meta, embeds)

if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description='Input Directory Path'
    )
    parser.add_argument('-meta', type=str,
                        help='Path to fold metadata')
    parser.add_argument('-embeds', type=str,
                        help='Path to directory with embeddings files')
    args = parser.parse_args()
    main(args.meta, args.embeds)

