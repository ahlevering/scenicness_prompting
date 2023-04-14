import yaml
import pickle
from pathlib import Path
from codebase.utils.file_utils import load_csv

import clip
import torch
import numpy as np
from pytorch_lightning.utilities.seed import seed_everything

from codebase.pt_funcs.dataloaders import PP2RankingsData, ClipDataLoader, PP2DataContainer
from codebase.experiment_tracking import process_yaml
    
##### SET GLOBAL OPTIONS ######
seed_everything(113)
np.set_printoptions(suppress=True)
torch.set_printoptions(sci_mode=False)

ratings = "data/votes.csv"
# Path(ratings).exists()
all_votes = load_csv(ratings)

setup_file = "setup_files/test/pp2_few_shot_multiprompt.yaml"
with open(setup_file) as file:
    exp_params = yaml.full_load(file)
out_dir = "data/embeddings/"
Path(out_dir).mkdir(exist_ok=True, parents=True)
data_container = PP2DataContainer(exp_params['paths']['labels_file'])

for architecture in ["ViT-L/14"]:
    model, preprocess = clip.load(architecture)
    model.requires_grad_ = False
    model.to(exp_params['hyperparams']['gpu_nums'][0])

    ##### SET UP TRANSFORMS #####
    test_trans = process_yaml.setup_transforms(exp_params['transforms']['test'])
    transforms = {'test': test_trans}

    ##### SETUP LOADERS #####
    data_module = ClipDataLoader(   12,  
                                    128,
                                    data_class=PP2RankingsData
                                    )

    # def setup_data_classes(self, splits_file, imgs_root, sample_files=None, embeddings=None, transforms=None, id_col='ID', splits=['train']):
    data_module.setup_data_classes( data_container,
                                    exp_params['paths']['images_root'],
                                    None,
                                    embeddings=None, # exp_params['paths']['embeddings'],
                                    transforms=transforms,
                                    id_col=exp_params['descriptions']['id_col'],
                                    splits=exp_params['descriptions']['splits']
                                )

    embeddings = {}
    with torch.no_grad():
        for batch in iter(data_module.test_dataloader()):
            # batch['img'] = batch['img'].cpu() #.to(exp_params['hyperparams']['gpu_num'])
            
            try:
                ### Left images ###
                to_encode = []
                for i, point in enumerate(batch['point_id_left']):
                    if not point in embeddings:
                        to_encode.append(i)
                imgs_to_encode = torch.stack([batch['img']['img_left'][i] for i in to_encode]).to(exp_params['hyperparams']['gpu_nums'][0])
                encodings = model.encode_image(imgs_to_encode)
                ids = [batch['point_id_left'][i] for i in to_encode]
                for i, id in enumerate(ids):
                    embeddings[id] = encodings[i].cpu()
            except Exception as e:
                print(e)

            try:
                ### Right images ###
                to_encode = []
                for i, point in enumerate(batch['point_id_right']):
                    if not point in embeddings:
                        to_encode.append(i)
                imgs_to_encode = torch.stack([batch['img']['img_left'][i] for i in to_encode]).to(exp_params['hyperparams']['gpu_nums'][0])
                encodings = model.encode_image(imgs_to_encode)
                ids = [batch['point_id_right'][i] for i in to_encode]
                for i, id in enumerate(ids):
                    embeddings[id] = encodings[i].cpu()
            except Exception as e:
                print(e)            

        archi_save_name = architecture.replace("/", "-") # Why did they use slashes in their naming!?
        with open(f"{out_dir}{archi_save_name}_pp2.pkl", 'wb') as f:
            pickle.dump(embeddings, f)