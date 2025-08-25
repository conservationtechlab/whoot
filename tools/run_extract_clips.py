from extract_noise import main
import argparse
import os
import yaml


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(
        description='Path to config file.'
    )
    PARSER.add_argument('-config', type=str,
                        help='Path to config.')
    ARGS = PARSER.parse_args()
    with open(ARGS.config, 'r', encoding='UTF-8') as f:
        config = yaml.safe_load(f)
    all_files = os.listdir(config['audio'])
    out = config['out']
    for file in all_files:
        try:
            print(f"running {file}")
            main(os.path.join(config['audio'], file), out)
        except Exception as e:
            print(f"couldnt load {file} because {e}")
