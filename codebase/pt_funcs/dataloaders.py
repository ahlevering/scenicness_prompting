from pathlib import Path
import pickle
import csv
from sklearn.model_selection import KFold

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
    
    def get_image(self, folder_num, id):
        img_path = Path(self.images_root + f'{folder_num}/' + str(id) +'.jpg')
        # img_path = Path(self.images_root + f'{datapoint["folder_num"].values[0]}/' + str(point_id) +'.jpg')
        img = Image.open(img_path).convert('RGB')
        img = self.transforms(img) # To list to append neighbouring images
        return img

    def __len__(self):
        if self.sample_ids:
            length = len(self.exp_data.labels[self.exp_data.labels['ID'].isin(self.sample_ids)])
        else:
            length = len(self.exp_data.labels)
        return length

class SoNDataContainer():
    def __init__(self, splits_file):
        self.labels = gpd.read_file(splits_file)

# class PP2Dataset(Dataset):
#     """Places Pulse 2 dataset."""

#     def __init__(self, data_path, sample_id_list=None, tr=transforms.ToTensor()):
#         im_paths_left = []
#         im_paths_right = []
#         winners = []
#         questions = []
#         question_names = ['lively', 'depressing', 'boring', 'beautiful', 'safety', 'wealthy']
#         winner_names = ['left', 'equal', 'right']

#         with open(os.path.join(data_path, 'metadata', 'final_data.csv')) as csvfile:
#             csvreader = csv.reader(csvfile)
#             headers = next(csvreader)
#             for row in csvreader:
#                 im_paths_left.append(os.path.join(data_path, 'final_photo_dataset', row[0] + '.jpg'))
#                 im_paths_right.append(os.path.join(data_path, 'final_photo_dataset', row[1] + '.jpg'))
#                 winners.append(row[2])
#                 questions.append(row[7])

#         for j in range(len(winners)):
#             for i in range(len(winner_names)):
#                 if winners[j] == winner_names[i]:
#                     winners[j] = i - 1
#         for j in range(len(questions)):
#             for i in range(len(question_names)):
#                 if questions[j] == question_names[i]:
#                     questions[j] = i

#         if sample_id_list is None:
#             self.im_paths_left = im_paths_left
#             self.im_paths_right = im_paths_right
#             self.winners = winners
#             self.questions = questions
#         else:
#             self.im_paths_left = [im_paths_left[i] for i in sample_id_list]
#             self.im_paths_right = [im_paths_right[i] for i in sample_id_list]
#             self.winners = [winners[i] for i in sample_id_list]
#             self.questions = [questions[i] for i in sample_id_list]
#         self.winner_names = winner_names
#         self.question_names = question_names
#         self.transform = tr
#         self.sample_id_list = sample_id_list

#     def __len__(self):
#         return len(self.im_paths_left)


#     def __getitem__(self, idx):
#         image_left = Image.open(self.im_paths_left[idx])
#         image_right = Image.open(self.im_paths_right[idx])
#         question = self.questions[idx]
#         winner = self.winners[idx]

#         image_left = self.transform(image_left)
#         image_right = self.transform(image_right)

#         sample = {'image_left': image_left, 'image_right': image_right, 'question': question, 'winner': winner}

#         return sample

class PP2RankingsData(Dataset):
    def __init__(self, exp_data, imgs_root, transforms, sample_ids=None, embeddings=None):    
        super().__init__()
        self.images_root = imgs_root
        self.transforms = transforms
        self.exp_data = exp_data
        self.sample_ids = sample_ids
        self.embeddings = embeddings

        self.scores = ["lively", "depressing" , "boring", "beautiful", "safety", "wealthy"]
        self.winner_scores = {'left':1.0, 'equal':0.5, 'right':0.0} # Target softmax confidence of left image

    def __getitem__(self, index):
        # Load labels
        if self.sample_ids:
            point_id = self.sample_ids[index]
            left_id, right_id = point_id.split("_")
            datapoint = self.exp_data.labels[(self.exp_data.labels['left_id'] == left_id) & (self.exp_data.labels['right_id'] == right_id)]
            datapoint = datapoint.iloc[0]
        else:
            datapoint = self.exp_data.labels.iloc[index]
        point_id_left = str(datapoint['left_id'])
        point_id_right = str(datapoint['right_id'])

        # Load embeddings or images
        if self.embeddings:
            img_left = self.embeddings[point_id_left]
            img_right = self.embeddings[point_id_right]
        else:
            img_left = self.get_image(point_id_left)
            img_right = self.get_image(point_id_right) # = self.get_image(point_id_left)
        
        # comparison_vector = [-1] * len(self.scores)
        # score_index = self.scores.index(self.embeddings['question'])
        q_score = self.winner_scores[datapoint['winner']]
        q_score_index = self.scores.index(datapoint['category'])

        ## Code for loading comparison
        return {'point_id_left': point_id_left,
                'point_id_right': point_id_right,
                'lat': float(datapoint['left_lat']),
                'lon': float(datapoint['left_long']),
                'img': {'img_left': img_left, 'img_right': img_right},
                'cat_name': datapoint['category'],
                'cat_index': q_score_index,
                'cat_score': q_score
            }

    def get_image(self, id):
        img_path = Path(self.images_root + str(id) +'.jpg')
        # img_path = Path(self.images_root + f'{datapoint["folder_num"].values[0]}/' + str(point_id) +'.jpg')
        img = Image.open(img_path).convert('RGB')
        img = self.transforms(img) # To list to append neighbouring images
        return img

    def __len__(self):
        if self.sample_ids:
            # length = len(self.exp_data.labels[(self.exp_data.labels['left_id'].isin(self.sample_ids))])
            length = len(self.sample_ids)
        else:
            length = len(self.exp_data.labels)
        return length
    
class PP2Data(Dataset):
    def __init__(self, exp_data, imgs_root, transforms, sample_ids=None, embeddings=None):    
        super().__init__()
        self.images_root = imgs_root
        self.transforms = transforms
        self.exp_data = exp_data
        self.sample_ids = sample_ids
        self.embeddings = embeddings

    def __getitem__(self, index):
        # Load labels
        if self.sample_ids:
            point_id = self.sample_ids[index]
            datapoint = self.exp_data.labels[self.exp_data.labels['Picture'] == point_id]
        else:
            datapoint = self.exp_data.labels.iloc[index]
            point_id = datapoint['Picture']

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
                'safe': float(datapoint['Safe']),
                'wealthy': float(datapoint['Wealthy'])
            }

    def __len__(self):
        if self.sample_ids:
            length = len(self.exp_data.labels[self.exp_data.labels['ID'].isin(self.sample_ids)])
        else:
            length = len(self.exp_data.labels)
        return length    

class PP2DataContainer():
    def __init__(self, ratings_file):
        labels = pd.read_csv(ratings_file)
        # self.labels = gpd.GeoDataFrame(labels, geometry=gpd.GeoSeries.from_xy(labels['Lon'], labels['Lat']), crs=4326)
        self.labels = pd.DataFrame(labels)

    def normalize(self, cols):
        # https://stackoverflow.com/questions/26414913/normalize-columns-of-a-dataframe
        x = self.labels[cols]
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        self.labels[cols] = pd.DataFrame(x_scaled)

class ClipDataLoader(pl.LightningDataModule):
    def __init__(self, n_workers, batch_size, data_class=SONData):
        super().__init__()
        self.batch_size = batch_size
        self.workers = n_workers
        self.data_class = data_class

        # self.dims = None    
        self.splits = None    

    def setup_data_classes(self, data_container, imgs_root, split_ids, embeddings=None, transforms=None, id_col='ID', splits=['train', 'val']):
        self.exp_data = data_container
        if embeddings:
            embeddings = load_pickle(embeddings)

        if 'all' in splits:
            self.test_data = self.data_class(self.exp_data,
                                             imgs_root,                                             
                                             transforms['test'],
                                             None,
                                             embeddings)

        if 'train' in splits:
            # train_split_labels = self.exp_data.labels[self.exp_data.labels['split'].isin(['train'])]
            # def __init__(self, exp_data, imgs_root, transforms, sample_ids=None, embeddings=None):            
            self.train_data = self.data_class(  self.exp_data,
                                                imgs_root,
                                                transforms['train'],
                                                split_ids['train'],
                                                embeddings)

        if 'val' in splits:
            # train_split_labels = self.exp_data.labels[self.exp_data.labels['split'].isin(['train'])]
            self.val_data = self.data_class(    self.exp_data,
                                                imgs_root,
                                                transforms['val'],
                                                split_ids['val'],
                                                embeddings)

        if 'test' in splits:
            self.test_data = self.data_class(self.exp_data,
                                             imgs_root,                                             
                                             transforms['test'],
                                             split_ids['test'],
                                             embeddings)
                                                
        self.splits = splits

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.workers, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=self.workers, shuffle=False)    

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=self.workers, shuffle=False)