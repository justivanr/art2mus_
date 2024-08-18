"""Script with shared constants for most scripts"""
import os
from pathlib import Path

# Directory in which the project is stored
PROJ_DIR = str(Path(__file__).resolve().parent.parent)

# Data folders
DATA_DIR = PROJ_DIR + "/data/"
ARTGRAPH_DIR = DATA_DIR + "images/imagesf2/"
FMA_DIR = DATA_DIR + "audio/fma_large/"
EXTRA_DATA_DIR = DATA_DIR + "/extra/"

# Scripts folders
SRC_DIR = PROJ_DIR + "/src/"
AUDIOLDM_DIR = SRC_DIR + "audioldm/"
AUDIOLDM2_DIR = SRC_DIR + "audioldm2/"
IMAGEBIND_DIR = SRC_DIR + "imagebind/"
SRC_DATA_DIR = SRC_DIR + "data/"