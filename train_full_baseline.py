import yaml

import torch
import numpy as np
from sklearn.model_selection import train_test_split
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
setup_file = "setup_files/train/son_full_baseline.yaml"
with open(setup_file) as file:
    exp_params = yaml.full_load(file)

labels_path, embeddings_path = get_label_embeds_paths(exp_params)
data_container = SoNDataContainer(labels_path, embeddings_path)

train_val_labels, test_labels = train_test_split(data_container.labels['ID'].values, test_size=0.1, random_state=113)
train_labels, val_labels = train_test_split(train_val_labels, test_size=0.15, random_state=113)

backbone = "ViT-L/14"
run_name = exp_params['descriptions']['name']
run_family = exp_params['descriptions']['exp_family']
k_folds = exp_params['hyperparams']['k_folds']

label_info = setup_label_info(["scenic"])

##### SET UP TRANSFORMS #####
transforms = setup_split_transforms(exp_params)

# prompts = exp_params['hyperparams']['prompts']
# prompts = torch.cat([clip.tokenize(p) for p in prompts])
# prompts = prompts.to(device=exp_params['hyperparams']['gpu_nums'][0])

############
# TRAINING #
############

storage_path = f"runs/{run_family}/{run_name}/full_baseline/"
split_ids = {'train': list(train_labels), 'val': list(val_labels), 'test': list(test_labels)}

##### STORE ENVIRONMENT AND FILES #####
organizer = organize_experiment_info(storage_path, setup_file)

##### SETUP DATASET #####
data_module = setup_data_module(SONData, exp_params, data_container, transforms, split_ids=split_ids, single_batch=False)

##### SETUP MODEL #####
net = setup_model(backbone, exp_params)
model = CLIPFewShotModule(organizer.root_path+'outputs/',
                            net,
                            run_name,
                            label_info,
                            exp_params['hyperparams']['use_precalc_embeddings'],
                            ['train', 'val'],
                            VarTrackerRegression)
model.set_hyperparams(exp_params['hyperparams']['optim']['lr'], exp_params['hyperparams']['optim']['decay'])

##### SETUP TRAINER #####
trainer = setup_trainer(organizer, run_name, exp_params, to_monitor="val_r2", monitor_mode='max')

##### FIT MODEL #####
trainer.fit(model, datamodule=data_module)

# ############################
# # RE-TRAIN FROM BEST STATE #
# ############################
# tb_logger = TensorBoardLogger(save_dir=organizer.logs_path, name=run_name)
# tester = Trainer(gpus=exp_params['hyperparams']['gpu_nums'], logger=tb_logger)
# tester.test(model, datamodule=data_module)