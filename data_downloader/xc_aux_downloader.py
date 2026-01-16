"""Downloads auxiliary Xeno-Canto data and audio files.

Relies on output from data_downloader/xc.py
Create a .env file with XC api-key
`XC_API_KEY=your_api_key_here`
Then call directly with `python xc_aux_downloader.py`
"""
import requests
import shutil
import os
import json
from pathlib import Path
from multiprocessing.pool import ThreadPool
from xc import XenoCantoDownloader
from dotenv import load_dotenv
import pandas as pd
import tqdm
import itertools


# https://stackoverflow.com/questions/16694907/download-large-file-in-python-with-requests
def download_file(url, local_filename, dry_run=False):
    if os.path.exists(local_filename):
        return local_filename

    try:
        with requests.get(url, stream=True) as r:
            with open(local_filename, 'wb') as f:
                if not dry_run:
                    shutil.copyfileobj(r.raw, f)
                else:
                    print(local_filename)

        return local_filename
    except IOError as e:
        print(e, flush=True)
        return None

def download_files(xcd, data, parent_folder="data/xeno-canto_aux", workers = 4):
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
    # Load environment variables from the .env file
    load_dotenv()

    xcd = XenoCantoDownloader(api_key=os.environ["XC_API_KEY"])

    with open("data/xc_meta.json", mode="r") as f:
        data = json.load(f)

    species = { recording["en"] for page in data for recording in page["recordings"] }

    # DEBUG
    # len({recording["en"] for page in data for recording in page["recordings"] })
    # len(species)

    data = []
    for specie in tqdm.tqdm(list(species)):
        data.append(xcd(query=f'en:"{specie}"'))

    data = list(itertools.chain.from_iterable(data))

    with open("xc_meta_aux.json", mode="w") as f:
        json.dump(data, f, indent=4)
        results = download_files(xcd, data)
        results

    recordings = xcd.concat_recording_data(data)
    df = pd.DataFrame(recordings)

    print(df.shape)