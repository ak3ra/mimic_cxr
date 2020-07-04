import pandas as pd
import os
from pathlib import Path

def create_img_list(data_path):
    '''
    function to loop through directory of mimic data and add each jpg to a list
    '''
    img_files = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith(".jpg"):
                img_files.append(os.path.join(root, file))
    return img_files
