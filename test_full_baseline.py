import yaml

import torch
import numpy as np
from sklearn.model_selection import train_test_split

from codebase.pt_funcs.dataloaders import SoNDataContainer, SONData
from codebase.pt_funcs.models_few_shot import BaselineModel
from codebase.experiment_tracking.run_tracker import VarTrackerRegression
from transformers import CLIPModel
from codebase.pt_funcs.models_baseline import RegressionModel

from codebase.exp_wrappers import *
    
##### SET GLOBAL OPTIONS ######
torch.manual_seed(113)
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

run_name = exp_params['descriptions']['name']
run_family = exp_params['descriptions']['exp_family']

label_info = setup_label_info(["scenic"])

##### SET UP TRANSFORMS #####
transforms = setup_split_transforms(exp_params)
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
model = RegressionModel(model.vision_model)
for _, param in model.vision_model._parameters.items(): 
    param.requires_grad = False

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

model = BaselineModel(organizer.root_path+'outputs/',
                            model,
                            run_name,
                            label_info,
                            exp_params['hyperparams']['use_precalc_embeddings'],
                            ['test'],
                            VarTrackerRegression)
model.set_hyperparams(exp_params['hyperparams']['optim']['lr'], exp_params['hyperparams']['optim']['decay'])

##### SETUP TRAINER #####
trainer = setup_trainer(organizer, run_name, exp_params, to_monitor="val_r2", monitor_mode='max')

##### FIT MODEL #####

# ############################
# # RE-TRAIN FROM BEST STATE #
# ############################

best_path = "runs/baselines/baseline_full_ViT/full_baseline/outputs/states/epoch=01-val_r2=0.8636.ckpt"
model.load_state_dict(torch.load(best_path)['state_dict'], strict=True)

############################
# RE-TRAIN FROM BEST STATE #
############################
tb_logger = WandbLogger(save_dir=organizer.logs_path, name=run_name)
tester = Trainer(accelerator="cuda", logger=tb_logger)
tester.test(model, datamodule=data_module)