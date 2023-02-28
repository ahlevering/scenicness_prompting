import yaml
import pickle
from pathlib import Path

import clip
import torch
import numpy as np
from pytorch_lightning.utilities.seed import seed_everything

from codebase.pt_funcs.dataloaders import SONData, ClipDataLoader, SoNDataContainer
from codebase.experiment_tracking import process_yaml
    
##### SET GLOBAL OPTIONS ######
seed_everything(113)
np.set_printoptions(suppress=True)
torch.set_printoptions(sci_mode=False)

setup_file = "setup_files/test/son_zero_shot.yaml"
with open(setup_file) as file:
    exp_params = yaml.full_load(file)
out_dir = "data/embeddings/"
Path(out_dir).mkdir(exist_ok=True, parents=True)

data_container = SoNDataContainer(exp_params['paths']['labels_file'])

for architecture in ["RN50", "ViT-L/14"]:
    model, preprocess = clip.load(architecture)
    model.requires_grad_ = False

    ##### SET UP TRANSFORMS #####
    test_trans = process_yaml.setup_transforms(exp_params['transforms']['test'])
    transforms = {'test': test_trans}

    ##### SETUP LOADERS #####
    data_module = ClipDataLoader(   16,  
                                    128,
                                    data_class=SONData
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
            encoding = model.encode_image(batch['img'].to(exp_params['hyperparams']['gpu_num']))
            for i, id_num in enumerate(batch['ids'].cpu().numpy()):
                embeddings[str(id_num)] = encoding[i,:].detach().cpu()

        archi_save_name = architecture.replace("/", "-") # Why did they use slashes in their naming!?
        with open(f"{out_dir}{archi_save_name}.pkl", 'wb') as f:
            pickle.dump(embeddings, f)
