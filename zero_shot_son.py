from copy import copy
import yaml
from pathlib import Path

import clip
import torch
import numpy as np
from torch import nn
from torchvision.models import resnet50
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.seed import seed_everything

from codebase.pt_funcs.dataloaders import SoNDataContainer, SONData, ClipDataLoader
from codebase.pt_funcs.models_few_shot import CLIPFewShotModule
from codebase.pt_funcs.models_zero_shot import PP2ManyPrompts
from codebase.utils.file_utils import load_csv, make_crossval_splits

from codebase.experiment_tracking import process_yaml
from codebase.experiment_tracking.save_metadata import ExperimentOrganizer
from codebase.experiment_tracking.run_tracker import VarTrackerClassification
    
##### SET GLOBAL OPTIONS ######
seed_everything(113)
np.set_printoptions(suppress=True)
torch.set_printoptions(sci_mode=False)

##### LOAD SET-UP FILE #####
setup_file = "setup_files/test/son_zero_shot.yaml"
with open(setup_file) as file:
    exp_params = yaml.full_load(file)

data_container = SoNDataContainer(exp_params['paths']['labels_file'])    

architecture = "ViT-L/14"
archi_save_name = architecture.replace("/", "-") # Why did they use slashes in their naming!?

run_name = exp_params['descriptions']['name']
run_family = exp_params['descriptions']['exp_family']

##### SETUP MODEL #####
##### Encode prompts #####
prompts = list(exp_params['hyperparams']['coop']['prompts'].keys())
prompts = torch.cat([clip.tokenize(p+".") for p in prompts])
prompts = prompts.to(device=exp_params['hyperparams']['gpu_nums'][0])
net, preprocess = clip.load(architecture)

values = list(exp_params['hyperparams']['coop']['prompts'].values())
values = torch.tensor(values, requires_grad=False).to(device=exp_params['hyperparams']['gpu_nums'][0])

net = PP2ManyPrompts(net, prompts, values, use_embeddings=True, son_rescale=True)

label_info = {}
for label in ["scenic"]:
    label_info[label] = {}
    label_info[label]['index'] = 0
    label_info[label]['ylims'] = [0, 1]


with torch.no_grad():
        ##### STORE ENVIRONMENT AND FILES #####
    base_path = f"runs/{run_family}/{run_name}"
    organizer = ExperimentOrganizer(base_path)
    organizer.store_yaml(setup_file)
    organizer.store_environment()
    organizer.store_codebase(['.py'])

    model = CLIPFewShotModule(organizer.root_path+'/outputs/', net, run_name, label_info, ['all'], VarTrackerClassification)

    # all_split_ids = load_csv(exp_params['paths']['splits_root']+f'{n_samples}.csv')
    # Subdivide sampled indices into k-fold bins
    ##### TEST MODEL #####
    ## Enable loading of embeddings
    data_module = ClipDataLoader(   8,    
                                    64,
                                    data_class=SONData
                                ) 

    data_module.setup_data_classes( 
                                    data_container,
                                    exp_params['paths']['images_root'],
                                    None,
                                    embeddings=exp_params['paths']['embeddings_root']+f"{archi_save_name}.pkl",
                                    transforms={'test':None},
                                    id_col=exp_params['descriptions']['id_col'],
                                    splits=['all'] 
                                )
    
    model.net.use_embeddings = True

    for i, param in enumerate(model.parameters()):
        param.requires_grad_(False)    
    tb_logger = TensorBoardLogger(save_dir=organizer.logs_path, name=run_name)
    tester = Trainer(gpus=exp_params['hyperparams']['gpu_nums'], logger=tb_logger)
    tester.test(model, datamodule=data_module)