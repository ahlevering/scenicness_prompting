import yaml

import torch
import numpy as np
import pandas as pd

from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.seed import seed_everything

from codebase.pt_funcs.dataloaders import SoNDataContainer, SONData
from codebase.pt_funcs.models_few_shot import CLIPFewShotModule
from codebase.experiment_tracking.run_tracker import VarTrackerRegression

from codebase.exp_wrappers import *
    
##### SET GLOBAL OPTIONS ######
seed_everything(113)
np.set_printoptions(suppress=True)
torch.set_printoptions(sci_mode=False)

##### LOAD SET-UP FILE #####
# setup_file = "setup_files/test/son_zero_shot_multiprompt.yaml"
setup_file = "setup_files/test/son_coop_contrastive_zero_shot.yaml"


with open(setup_file) as file:
    exp_params = yaml.full_load(file)
    
all_prompts = pd.read_csv("data/prompts_cleaned.csv")
prompts = {}
for i, key in enumerate(all_prompts['prompts_in_order'].values):
    prompts[key] = all_prompts['values_in_order'].iloc[i]
exp_params['hyperparams']['prompts'] = prompts

labels, embeddings = get_label_embeds_paths(exp_params)
data_container = SoNDataContainer(labels, embeddings)

backbone = "ViT-L/14"
run_name = exp_params['descriptions']['name']
run_family = exp_params['descriptions']['exp_family']

label_info = setup_label_info(["scenic"])

##### SETUP MODEL #####
##### Encode prompts #####

transforms = {'test': None}

with torch.no_grad():
    ##### STORE ENVIRONMENT AND FILES #####
    base_path = f"runs/{run_family}/{run_name}/"
    organizer = organize_experiment_info(base_path, setup_file)

    data_module = setup_data_module(SONData, exp_params, data_container, transforms, split_ids=None, single_batch=False)

    net = setup_model(backbone, exp_params)
    model = CLIPFewShotModule(organizer.root_path+'outputs/',
                              net,
                              run_name,
                              label_info,
                              exp_params['hyperparams']['use_precalc_embeddings'],
                              ['all'],
                              VarTrackerRegression)

    for i, param in enumerate(model.parameters()):
        param.requires_grad_(False)    
    tb_logger = TensorBoardLogger(save_dir=str(organizer.logs_path))
    tester = Trainer(gpus=exp_params['hyperparams']['gpu_nums'], logger=tb_logger)
    tester.test(model, datamodule=data_module)

# net, preprocess = clip.load(architecture)

# values = list(exp_params['hyperparams']['coop']['prompts'].values())
# values = torch.tensor(values, requires_grad=False).to(device=exp_params['hyperparams']['gpu_nums'][0])

# net = PP2ManyPrompts(net, prompts, values, use_embeddings=True, son_rescale=True)    

# prompts = list(exp_params['hyperparams']['coop']['prompts'].keys())
# prompts = torch.cat([clip.tokenize(p+".") for p in prompts])
# prompts = prompts.to(device=exp_params['hyperparams']['gpu_nums'][0])