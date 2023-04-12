import yaml

import torch
import numpy as np

from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.seed import seed_everything

from codebase.pt_funcs.dataloaders import PP2DataContainer, PP2RankingsData
from codebase.pt_funcs.models_few_shot import CLIPComparisonModule
from codebase.experiment_tracking.run_tracker import VarTrackerClassification

from codebase.exp_wrappers import *
    
##### SET GLOBAL OPTIONS ######
seed_everything(113)
np.set_printoptions(suppress=True)
torch.set_printoptions(sci_mode=False)

##### LOAD SET-UP FILE #####
setup_file = "setup_files/train/pp2_zero_shot.yaml"
with open(setup_file) as file:
    exp_params = yaml.full_load(file)
    
labels, embeddings = get_label_embeds_paths(exp_params)
data_container = PP2DataContainer(labels, embeddings)

backbone = "ViT-L/14"
run_name = exp_params['descriptions']['name']
run_family = exp_params['descriptions']['exp_family']

label_info = setup_pp2_label_info()

##### SETUP MODEL #####

transforms = {'test': None}

with torch.no_grad():
    ##### STORE ENVIRONMENT AND FILES #####
    base_path = f"runs/{run_family}/{run_name}/"
    organizer = organize_experiment_info(base_path, setup_file)

    data_module = setup_data_module(PP2RankingsData, exp_params, data_container, transforms, split_ids=None, single_batch=False)

    net = setup_model(backbone, exp_params)
    model = CLIPComparisonModule(organizer.root_path+'outputs/',
                                net,
                                run_name,
                                label_info,
                                exp_params['hyperparams']['use_precalc_embeddings'],
                                ['all'],
                                VarTrackerClassification)

    for i, param in enumerate(model.parameters()):
        param.requires_grad_(False)    
    tb_logger = TensorBoardLogger(save_dir=str(organizer.logs_path))
    tester = Trainer(gpus=exp_params['hyperparams']['gpu_nums'], logger=tb_logger)
    tester.test(model, datamodule=data_module)