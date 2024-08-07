# script to download images from Geograph using SoN tsv file

# import libraries

import os
import logging
import glob
import re
import time
import requests
import pandas as pd
from pathlib import Path
from PIL import Image
from io import BytesIO
from tqdm import tqdm

def photo_download(son_csv, write_location):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()        

    son_csv = pd.read_csv(son_csv, sep='\t')
    list_of_files = glob.glob(f'{write_location}*') # * means all if need specific format then *.jpg

    if len(list_of_files) == 0: # check progress, download next image
        start = 0

    else:
        latest_file = max(list_of_files, key=os.path.getctime)

        id = int(re.findall(r'(\d+(?=\.))', latest_file)[0])
        start = son_csv.loc[son_csv['ID'] == id].index[0] + 1

    for idx in tqdm(range(start, len(son_csv))):
        img_server_id = son_csv.iloc[idx]['Geograph URI'].split('/')[-1]

        url = f"https://t0.geograph.org.uk/stamp.php?id={img_server_id}&title=on&gravity=SouthEast&hash=ce524193"
        response = requests.get(url)

        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
            image.save(f'{write_location}{idx}.jpg') # Save with row index rather than Geograph server index 
        else:
            logger.error(f'Failed to retrieve Geograph img ID {img_server_id}')

        time.sleep(0.1) # Rate limiting. Please be nice to the good folks at Geograph.

if __name__ == '__main__':   
    son_csv = 'data/votes.tsv'
    out_dir = 'data/images/'

    Path(out_dir).mkdir(exist_ok=True, parents=True)
    photo_download(son_csv, out_dir)
