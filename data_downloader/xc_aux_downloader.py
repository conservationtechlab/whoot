"""Downloads auxiliary Xeno-Canto data and audio files.

Relies on output from data_downloader/xc.py
Create a .env file with XC api-key
`XC_API_KEY=your_api_key_here`
Then call directly with `python xc_aux_downloader.py`
"""

import shutil
import os
import json
import itertools
from pathlib import Path
from multiprocessing.pool import ThreadPool
from dotenv import load_dotenv
import pandas as pd
import tqdm
import requests
from xc import XenoCantoDownloader


# https://stackoverflow.com/questions/16694907/download-large-file-in-python-with-requests
def download_file(url, local_filename, dry_run=False):
    """Download a file from a url to a local file.

    Args:
        url (str): url to download file from
        local_filename (str): path to local file to save to
        dry_run (bool): if True, do not actually download file
    Returns:
        local_filename (str): path to local file or None if failed
    """
    if os.path.exists(local_filename):
        return local_filename

    try:
        with requests.get(url, stream=True, timeout=1000) as r:
            with open(local_filename, 'wb') as f:
                if not dry_run:
                    shutil.copyfileobj(r.raw, f)
                else:
                    print("Pretend download of", local_filename)

        return local_filename
    except IOError as e:
        print(e, flush=True)
        return None


def download_files(
        xcd: XenoCantoDownloader,
        data: list,
        parent_folder: str = "data/xeno-canto_aux",
        workers: int = 4
):
    """Download all the files collected by the Xeno-Canto downloader.

    Args:
        xcd (XenoCantoDownloader): the Xeno-Canto downloader object
            Allows for preprocessing of recording metadata
        data (list): list of recording data dicts
        parent_folder (str): path to folder to store audio files
        workers (int): number of parallel download workers
            Tune down if hitting rate limits
    Returns:
        results (list): list of downloaded file paths
    """
    def prep_download(args):
        url = args[0]
        file_path = args[1]
        return download_file(url, file_path)

    os.makedirs(parent_folder, exist_ok=True)

    if "recordings" in data[0]:
        data = xcd.concat_recording_data(data)
    download_data = [
        (recording["file"], Path(parent_folder) / Path(recording["file-name"]))
        for recording in data
    ]
    pool = ThreadPool(workers)
    results = pool.imap_unordered(prep_download, download_data)
    pool.close()
    return results


def main():
    """Script to download auxiliary Xeno-Canto data and audio files."""
    # Load environment variables from the .env file
    load_dotenv()

    xcd = XenoCantoDownloader(api_key=os.environ["XC_API_KEY"])

    with open("data/xc_meta.json", mode="r", encoding="utf-8") as f:
        data = json.load(f)

    species = {
        recording["en"] for page in data for recording in page["recordings"]
    }

    data = []
    for specie in tqdm.tqdm(list(species)):
        data.append(xcd(query=f'en:"{specie}"'))

    data = list(itertools.chain.from_iterable(data))

    with open("xc_meta_aux.json", mode="w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
        results = download_files(xcd, data)
        print("Done downloading files, num downloaded:", len(results))

    recordings = xcd.concat_recording_data(data)
    df = pd.DataFrame(recordings)

    print("Metadata has shape:", df.shape)


if __name__ == "__main__":
    main()
