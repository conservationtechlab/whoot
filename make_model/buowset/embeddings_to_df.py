



def obtain_perch_embeddings(embeds):
    """Create dict dataframe with filename and embedding list
    """
    # placeholder for actual function
    embeddings_df = embeds

    return embeddings_df


def obtain_birdnet_embeddings(embeds):
    """Create a dict dataframe with filename and embedding list

    Args:
        embeds (str): Path to directory where embeddings files are.

    Returns:
        embed_df (pd.Dateframe): A dictonary with the filename as the
                                 key and the list of floats (embeddings)
                                 as the value
    """
    embed_dict = {}
    text_files = glob.glob(os.path.join(embeds, "*.txt"))
    for embed in text_files:
        filename = ntpath.basename(embed)
        filename = filename.replace(".birdnet.embeddings.txt", ".wav")
        dfb = pd.read_csv(embed,
                          delimiter="[,\t]",
                          engine='python',
                          header=None)
        dfb_stripped = dfb.drop(dfb.columns[:2], axis=1)
        flattened = dfb_stripped.values.flatten()
        if len(flattened) > 1024:
            print(f"filename {filename} has extra lines. Trunicating")
            flattened = flattened[:1024]
        embed_dict[filename] = flattened

    embed_df = pd.DataFrame({
        'filename': list(embed_dict.keys()),
        'embeddings': list(embed_dict.values())
    })

    return embed_df


def main(embeds, source, path):
    """
    """
    if source == 'birdnet':
        embed_df = obtain_birdnet_embeddings(embeds)
    elif source == 'perch':
        embed_df = obtain_perch_embeddings(embeds)
    else:
        print("cannot obtain embeddings, check source")

    embed_df.to_csv(path, encoding='utf-8', index=False)
    print(f"Created dataframe file: {path}")


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(
        description='Input Directory Path'
    )
    PARSER.add_argument('-embeds', type=str,
                        help='Path to directory containing embedding info.')
    PARSER.add_argument('-source', type=str,
                        help='birdnet or perch (for now)')
    PARSER.add_argument('-path', type=str,
                        help='Path to output dataframe')
    ARGS = PARSER.parse_args()
    main(ARGS.embeds, ARGS.source, ARGS.path)

