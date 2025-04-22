"""
"""
import pandas as pd
import argparse

def create_strat_folds(df):
    """
    """
    obtain complete class distribution for all classes
    divide each class by 5, thats how many need to be in each fold
    lets say theres 5000 bo buow, 1000 cluck, 500 coocoo, 500 chick begging, 2000 alarm, 1000 twitter
    so each fold needs, 1000 no buow, 100 coocoo, 100 chick begging, 400 alarm, 200 twitter
    
    if i create a df that has the wav path, the distribution of calls in a column. i can randomly add different ones
    together until i get about that disribution per class. then i can go in and add a column to the df that has the fold it's in. donezo
    what combination of these properites will give me the closest distribution to a balanced set? 

def main(meta):
    """
    """
    df = pd.read(meta)
    create_strat_folds(df)


if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description='Input Directory Path'
        )
    parser.add_argument('meta', type=str,
                        help='Path to metadata csv')
    args = parser.parse_args()
    main(args.meta)

