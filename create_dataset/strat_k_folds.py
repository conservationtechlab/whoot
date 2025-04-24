"""
"""
import pandas as pd
import argparse

def create_strat_folds(df):
    """
    """
    cluck_df = df[df['label'] == 'cluck']
    coocoo_df = df[df['label'] == 'coocoo']
    twitter_df = df[df['label'] == 'twitter']
    alarm_df = df[df['label'] == 'alarm']
    chick_beg_df = df[df['label'] == 'chick_begging']
    no_buow_df = df[df['label'] == 'no_buow']
    # amount needed per fold
    print(f"{chick_beg_df}")
    len_cluck = len(cluck_df) / 5
    len_coocoo = len(coocoo_df) / 5
    len_twitter = len(twitter_df) / 5
    len_alarm = len(alarm_df) / 5
    len_chick = len(chick_beg_df) / 5
    len_no_buow = len(no_buow_df) / 5
    print(f"len_cluck: {len_cluck} len_coocoo: {len_coocoo} len_twitter: {len_twitter} len_alarm: {len_alarm} len_chick: {len_chick} len_no_buow: {len_no_buow}")
    

    grouped = df.groupby('original_path')
    for index, group in grouped:
        print(f"group {index}")
        print(group)
    '''if i create a df that has the wav path, the distribution of calls in a column. i can randomly add different ones
    together until i get about that disribution per class. then i can go in and add a column to the df that has the fold it's in. donezo
    what combination of these properites will give me the closest distribution to a balanced set?''' 

def main(meta):
    """
    """
    df = pd.read_csv(meta)
    create_strat_folds(df)


if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description='Input Directory Path'
        )
    parser.add_argument('meta', type=str,
                        help='Path to metadata csv')
    args = parser.parse_args()
    main(args.meta)

