"""Create embedding dataframe as well as SVM


"""
from embed_to_df_birdnet import obtain_birdnet_embeddings
from make_svm import get_binary_classes, save_out_model


def main(embed_path, meta, model):
    """
    """
    metadata = pd.read_csv(meta, index_col=0)
    df_merged = obtain_birdnet_embeddings(metadata, embed_path)
    dataset = get_binary_classes(df_merged)
    if model is None:
        make_svm(dataset)
    else:
        svm = make_svm(dataset)
        save_out_model(svm, model)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(
        description='Input Directory Path'
    )
    PARSER.add_argument('-meta', type=str,
                        help='Path to metadata with fold and label info.')
    PARSER.add_argument('-embed_path', type=str,
                        help='Path to directory containing embedding info.')
    PARSER.add_argument('-model', type=str, default=None,
                        help='Path to output dataframe as .pkl')
    ARGS = PARSER.parse_args()
    main(ARGS.meta, ARGS.embed_path, ARGS.model)

