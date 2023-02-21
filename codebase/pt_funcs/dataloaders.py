from pathlib import Path
import pickle
import csv

import pandas as pd
import geopandas as gpd
from sklearn import preprocessing
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

def load_pickle(pickle_file):
    with open(pickle_file, 'rb') as f:
        matches = pickle.load(f)
    return matches

def load_csv(fpath):
    with open(fpath, 'r', newline='') as csv_file:
        f = csv.reader(csv_file, quoting=csv.QUOTE_ALL)
        rows = [row for row in f][0]
        rows = [int(r) for r in rows] # Fix types
        return rows

class SONData(Dataset):
    def __init__(self, exp_data, imgs_root, transforms, sample_ids=None, embeddings=None):
        self.images_root = imgs_root
        
        self.transforms = transforms
        self.exp_data = exp_data
        self.sample_ids = sample_ids
        self.embeddings = embeddings

    def __getitem__(self, index):
        # Load labels
        if self.sample_ids:
            point_id = self.sample_ids[index]
            datapoint = self.exp_data.labels[self.exp_data.labels['ID'] == point_id].squeeze()
        else:
            datapoint = self.exp_data.labels.iloc[index]
            point_id = datapoint['ID']

        if self.embeddings:
            img = self.embeddings[str(point_id)]
        else:
            img = self.get_image(datapoint["folder_num"], point_id)

        scenicness = float(datapoint['Average'])

        return {'ids': point_id,
                'lat': float(datapoint['Lat']),
                'lon': float(datapoint['Lon']),
                'img': img,
                'gt': scenicness
            }
    
    def get_image(self, id, folder_num):
        img_path = Path(self.images_root + f'{folder_num}/' + str(id) +'.jpg')
        # img_path = Path(self.images_root + f'{datapoint["folder_num"].values[0]}/' + str(point_id) +'.jpg')
        img = Image.open(img_path).convert('RGB')
        img = self.transforms(img) # To list to append neighbouring images

    def __len__(self):
        if self.sample_ids:
            length = len(self.exp_data.labels[self.exp_data.labels['ID'].isin(self.sample_ids)])
        else:
            length = len(self.exp_data.labels)
        return length

class SoNDataContainer():
    def __init__(self, splits_file):
        self.labels = gpd.read_file(splits_file)

class PP2Data(Dataset):
    def __init__(self, exp_data, imgs_root, transforms) -> None:
        super().__init__()
        self.images_root = imgs_root
        self.transforms = transforms

        self.exp_data = exp_data

    def __getitem__(self, index):
        # Load labels
        point_id = self.exp_data.iloc[index]
        datapoint = self.exp_data.labels[self.exp_data.labels['Picture'] == point_id]

        img_path = Path(self.images_root + str(point_id) +'.jpg')
        # Load image
        img = Image.open(img_path).convert('RGB')
        img = self.transforms(img) # To list to append neighbouring images

        return {'ids': point_id,
                'lat': float(datapoint['Lat']),
                'lon': float(datapoint['Lon']),
                'img': img,
                'lively': float(datapoint['Lively']),
                'depressing': float(datapoint['Depressing']),
                'boring': float(datapoint['Boring']),
                'beautiful': float(datapoint['Beautiful']),
                'safe': float(datapoint['Safety']),
                'wealthy': float(datapoint['Wealthy'])
            }

    def __len__(self):
        return len(self.split_ids)

class PP2DataContainer():
    def __init__(self, splits_file):
        labels = pd.read_csv(splits_file)
        self.labels = gpd.GeoDataFrame(labels, geometry=gpd.GeoSeries.from_xy(labels['Lon'], labels['Lat']), crs=4326)

    def normalize(self, cols):
        # https://stackoverflow.com/questions/26414913/normalize-columns-of-a-dataframe
        x = self.labels[cols]
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        self.labels[cols] = pd.DataFrame(x_scaled)

class ClipDataLoader(pl.LightningDataModule):
    def __init__(self, n_workers, batch_size, data_class=SONData, container_class=SoNDataContainer):
        super().__init__()
        self.batch_size = batch_size
        self.workers = n_workers
        self.data_class = data_class
        self.container_class = container_class

        # self.dims = None    
        self.splits = None    

    def setup_data_classes(self, splits_file, imgs_root, sample_files=None, embeddings=None, transforms=None, id_col='ID', splits=['train']):
        self.exp_data = self.container_class(splits_file)
        if embeddings:
            embeddings = load_pickle(embeddings)

        if 'all' in splits:
            # exp_data, imgs_root, transforms, sample_ids=None, embeddings=None
            self.test_data = self.data_class(self.exp_data,
                                             imgs_root,                                             
                                             transforms['test'],
                                             sample_files,
                                             embeddings)

        if 'train' in splits:
            # train_split_labels = self.exp_data.labels[self.exp_data.labels['split'].isin(['train'])]
            train_sample_ids = load_csv(sample_files['train'])
            self.train_data = self.data_class(  self.exp_data,
                                                train_sample_ids,
                                                imgs_root,
                                                transforms=transforms['train'])
        self.splits = splits

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.workers, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=self.workers, shuffle=False)