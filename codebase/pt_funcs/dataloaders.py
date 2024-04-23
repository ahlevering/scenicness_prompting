from pathlib import Path

import pandas as pd
import geopandas as gpd
from sklearn import preprocessing
from PIL import Image
from PIL import ImageFile

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

from codebase.utils.file_utils import load_pickle, load_csv

ImageFile.LOAD_TRUNCATED_IMAGES = True

class SONData(Dataset):
    def __init__(self, exp_data, imgs_root, transforms, sample_ids=None, use_embeddings=False):
        self.images_root = imgs_root
        self.transforms = transforms
        self.exp_data = exp_data
        self.sample_ids = sample_ids
        self.use_embeddings = use_embeddings

    def __getitem__(self, index):
        # Load labels
        if self.sample_ids:
            point_id = self.sample_ids[index]
            datapoint = self.exp_data.labels[self.exp_data.labels['ID'] == point_id].squeeze()
        else:
            datapoint = self.exp_data.labels.iloc[index]
            point_id = datapoint['ID']

        if self.use_embeddings:
            img = self.exp_data.embeddings[str(point_id)]
        else:
            img = self.get_image(datapoint["folder_num"], point_id)

        scenicness = float(datapoint['Average'])

        return {'ids': point_id,
                'lat': float(datapoint['Lat']),
                'lon': float(datapoint['Lon']),
                'img': img,
                'gt': scenicness
            }
    
    def get_image(self, folder_num, id):
        img_path = Path(self.images_root + f'{folder_num}/' + str(id) +'.jpg')
        # img_path = Path(self.images_root + f'{datapoint["folder_num"].values[0]}/' + str(point_id) +'.jpg')
        img = Image.open(img_path).convert('RGB')
        if self.transforms:
            img = self.transforms(img) # To list to append neighbouring images
        return img

    def __len__(self):
        if self.sample_ids:
            length = len(self.exp_data.labels[self.exp_data.labels['ID'].isin(self.sample_ids)])
        else:
            length = len(self.exp_data.labels)
        return length

class SoNDataContainer():
    def __init__(self, splits_file, embeddings=None):
        self.labels = gpd.read_file(splits_file)
        if embeddings:
            self.embeddings = load_pickle(embeddings)
        else:
            self.embeddings = None

class ClipDataLoader(pl.LightningDataModule):
    def __init__(self, n_workers, batch_size, data_class=SONData):
        super().__init__()
        self.batch_size = batch_size
        self.workers = n_workers
        self.data_class = data_class

        # self.dims = None    
        self.splits = None    

    def setup_data_classes(self, data_container, imgs_root, split_ids, embeddings_policy, transforms=None, splits=['train', 'val']):
        self.exp_data = data_container

        if 'all' in splits:
            self.test_data = self.data_class(self.exp_data,
                                             imgs_root,                                      
                                             transforms['test'],
                                             None,
                                             embeddings_policy['test'])

        if 'train' in splits:
            self.train_data = self.data_class(self.exp_data,
                                              imgs_root,
                                              transforms['train'],
                                              split_ids['train'],
                                              embeddings_policy['train'])

        if 'val' in splits:
            self.val_data = self.data_class(self.exp_data,
                                            imgs_root,
                                            transforms['val'],
                                            split_ids['val'],
                                            embeddings_policy['val'])

        if 'test' in splits:
            self.test_data = self.data_class(self.exp_data,
                                             imgs_root,                                             
                                             transforms['test'],
                                             split_ids['test'],
                                             embeddings_policy['test'])
                                                
        self.splits = splits

    def collate_fn(self, batch):
        batch_out = {}
        for key in ['ids', 'lat', 'lon', 'gt']:
            batch_out[key] = torch.stack([torch.tensor(b[key]) for b in batch])
        batch_out['img'] = [b['img'] for b in batch] 
        return batch_out

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.workers, shuffle=True, collate_fn=self.collate_fn)
    
    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=self.workers, shuffle=False, collate_fn=self.collate_fn)    

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=self.workers, shuffle=False, collate_fn=self.collate_fn)