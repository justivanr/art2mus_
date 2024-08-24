"""
Script to download FMA Large Dataset from Kaggle.

You need to generate a token on Kaggle and use the values 
found in the .json file to instantiate the environmental variables below.
"""

import os
os.environ['KAGGLE_USERNAME'] = "justivanr"
os.environ['KAGGLE_KEY'] = "7a144ecaf3b17ed4497fb338d02e75ff"
import sys
sys.path.append("src")
import conf
from kaggle.api.kaggle_api_extended import KaggleApi

# Directory in which the project is stored
PROJ_DIR = conf.PROJ_DIR
DATA_FOLDER = PROJ_DIR + "/data/"
AUDIO_FOLDER = DATA_FOLDER + "audio/"
FMA_FOLDER = DATA_FOLDER + "audio/fma_large/"
dataset_name = "noahbadoa/fma-dataset-100k-music-wav-files"


if os.path.exists(FMA_FOLDER) and len(os.listdir(FMA_FOLDER)) == 158:
    download_data = False
    print(f"FMA dataset is already available! No need to download it!")
else:
    print(f"Downloading FMA dataset.....")
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(dataset=dataset_name, path=AUDIO_FOLDER, unzip=True)
    print(f"FMA dataset has been downloaded and is ready to be used!")