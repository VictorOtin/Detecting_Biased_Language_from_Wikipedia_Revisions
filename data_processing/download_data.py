import urllib.request
from pathlib import Path
import zipfile
import os


def download_file(url, path):
    Path(path).mkdir(parents=False, exist_ok=True)
    filename = url.split('/')[-1]
    path_to_file = path+'/'+filename
    if Path(path_to_file).is_file():
        print("File already exists: {}".format(path_to_file))
    else:
        urllib.request.urlretrieve(url, path_to_file)
        print("File downloaded: {}".format(path_to_file))
    return path_to_file


def remove_test():
    filepath = '../../data/raw/WPRH/full/temp.txt'
    os.remove(filepath)



def download_pryzant():
    print("This is going to download and unzip the data from the Pryzant2019 paper (100MB zipped, 500MB unzipped")
    print("Downloading...")
    # From https://github.com/rpryzant/neutralizing-bias

    url = 'http://bit.ly/bias-corpus'
    # DEV TEST
    #url = "https://file-examples.com/wp-content/uploads/2017/02/zip_2MB.zip"

    save_path = '../../data/raw/Pryzant2019/'
    # Create Pryzant2019 directory if it doesnt exist
    Path(save_path).mkdir(parents=False, exist_ok=True)
    urllib.request.urlretrieve(url, save_path+'pryzant2019_data.zip')

    print('Download complete, unzipping...')
    # Unzip
    with zipfile.ZipFile(save_path+'pryzant2019_data.zip','r') as zip_ref:
        zip_ref.extractall(save_path)

    # Delete zip file
    Path(save_path+'pryzant2019_data.zip').unlink()

    print('Unzipping complete. File is at {}'.format(save_path+'bias_data'))
    

