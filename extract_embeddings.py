import yaml
import pickle

import clip
import torch
import numpy as np
from pytorch_lightning.utilities.seed import seed_everything

from codebase.pt_funcs.dataloaders import SONData, ClipDataLoader
from codebase.experiment_tracking import process_yaml
    
##### SET GLOBAL OPTIONS ######
seed_everything(113)
np.set_printoptions(suppress=True)
torch.set_printoptions(sci_mode=False)

setup_file = "setup_files/test/son_zero_shot.yaml"
with open(setup_file) as file:
    exp_params = yaml.full_load(file)

model, preprocess = clip.load('ViT-B/32')
model.requires_grad_ = False
# model = model#.to(exp_params['hyperparams']['gpu_num']).eval()
# model = model.to(device=exp_params['hyperparams']['gpu_num'])

##### SET UP TRANSFORMS #####
test_trans = process_yaml.setup_transforms(exp_params['transforms']['test'])
transforms = {'test': preprocess}
# transforms = {'test': test_trans}

# label_info = {}
# label_info['scenic'] = {}
# # label_info['score']['index'] = 0
# label_info['scenic']['ylims'] = [1, 10]

##### SETUP LOADERS #####
data_module = ClipDataLoader( 16,  
                              64,
                              data_class=SONData
                            )

# def setup_data_classes(self, splits_file, imgs_root, sample_files=None, embeddings=None, transforms=None, id_col='ID', splits=['train']):
data_module.setup_data_classes( exp_params['paths']['splits_file'],
                                exp_params['paths']['images_root'],
                                None,
                                False,
                                transforms,
                                exp_params['descriptions']['id_col'],
                                exp_params['descriptions']['splits']
                            )

embeddings = {}
with torch.no_grad():
    for batch in iter(data_module.test_dataloader()):
        # batch['img'] = batch['img'].cpu() #.to(exp_params['hyperparams']['gpu_num'])
        encoding = model.encode_image(batch['img'].to(exp_params['hyperparams']['gpu_num']))
        for i, id_num in enumerate(batch['ids'].cpu().numpy()):
            embeddings[str(id_num)] = encoding[i,:].detach().cpu()

    with open("data/embeddings_preprocess.pkl", 'wb') as f:
        pickle.dump(embeddings, f)
