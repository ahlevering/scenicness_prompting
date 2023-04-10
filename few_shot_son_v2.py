import yaml
from pathlib import Path

import torch
import numpy as np

from pytorch_lightning.utilities.seed import seed_everything

from codebase.pt_funcs.dataloaders import SoNDataContainer, SONData, ClipDataLoader
from codebase.pt_funcs.models_few_shot import SONCLIPFewShotNet, CLIPFewShotModule, SONCLIPPromptSubsetter, Baseline
from codebase.utils.file_utils import load_csv, make_crossval_splits
from codebase.experiment_tracking.run_tracker import VarTrackerRegression
from codebase.experiment_tracking.save_metadata import ExperimentOrganizer

from codebase.pt_funcs.dataloaders import SONData

from codebase.exp_wrappers import *
    
##### SET GLOBAL OPTIONS ######
seed_everything(113)
np.set_printoptions(suppress=True)
torch.set_printoptions(sci_mode=False)

##### LOAD SET-UP FILE #####
# setup_file = "setup_files/train/son_few_shot.yaml"
setup_file = "setup_files/train/son_few_shot.yaml"
with open(setup_file) as file:
    exp_params = yaml.full_load(file)

data_container = SoNDataContainer(exp_params['paths']['labels_file'], exp_params['paths']['embeddings_file'])

backbone = "ViT-L/14"
# archi_save_name = architecture.replace("/", "-") # Why did they use slashes in their naming!?

run_name = exp_params['descriptions']['name']
run_family = exp_params['descriptions']['exp_family']
label_info = setup_son_label_info()

##### SET UP TRANSFORMS #####
transforms = setup_split_transforms(exp_params)

# prompts = exp_params['hyperparams']['prompts']
# prompts = torch.cat([clip.tokenize(p) for p in prompts])
# prompts = prompts.to(device=exp_params['hyperparams']['gpu_nums'][0])

############
# TRAINING #
############

k_folds = exp_params['hyperparams']['k_folds']
state = None
for n_samples in [500]:#[25, 50, 100, 250, 500]:
    if n_samples >= 250:
        min_epoch = 0
    else:
        min_epoch = 6 # Only load states from after the 6th epoch to avoid unstable models
    sample_split_ids = get_sample_split_ids(exp_params, n_samples, to_int=True)

    lr_rsquared = []
    for lr in exp_params['hyperparams']['optim']['lr']:
        run_file_storage_path = f"runs/{run_family}/{run_name}/{n_samples}/{lr}/"
        if k_folds > 1:
            k_rsquared = []
            for k in range(k_folds):
                train_ids, val_ids, test_ids = get_crossval_splits(sample_split_ids, k_folds, data_container)
                split_k_ids = {'train': train_ids[k], 'val': val_ids[k]}

                ##### STORE ENVIRONMENT AND FILES #####
                run_file_storage_path += f"/val_{k}/"
                organizer = organize_experiment_info(run_file_storage_path, setup_file)

                ##### SETUP DATASET #####
                data_module = setup_data_module(SONData, exp_params, data_container, transforms, split_ids=split_k_ids, single_batch=True)

                ##### SETUP MODEL #####
                net = setup_model(backbone, exp_params, use_embeddings=True, model_type="learned_prompt_context")
                model = CLIPFewShotModule(organizer.root_path+'outputs/', net, run_name, label_info, ['train', 'val'], VarTrackerRegression)
                model.set_hyperparams(lr, exp_params['hyperparams']['optim']['decay'])

                ##### SETUP TRAINER #####
                trainer = setup_trainer(organizer, run_name, exp_params, to_monitor="val_r2", monitor_mode='max')
                
                ##### FIT MODEL #####
                trainer.fit(model, datamodule=data_module)            
                k_rsquared.append(model.val_tracker.variables['scenic'].metrics['rsquared'][-1])

            # Aggregate metric over all k splits
            lr_rsquared.append(np.mean(k_rsquared))

############################
# RE-TRAIN FROM BEST STATE #
############################
    run_file_storage_path = f"runs/{run_family}/{run_name}/{n_samples}/{lr}/"
    if k_folds > 1:
        ##### GET BEST HYPERPARAMETER & STATE #####
        best_index = np.argmax(lr_rsquared)
        lr = exp_params['hyperparams']['optim']['lr'][best_index]

        best_states_dir = f"runs/{run_family}/{run_name}/{n_samples}/{lr}/val_{best_index}/"
        top_model_path = get_top_model(best_states_dir, min_epoch)
        state = torch.load(top_model_path)['state']
        run_file_storage_path += "best/"

    test_ids = data_container.labels[~data_container.labels['ID'].isin(sample_split_ids)]

    ##### STORE ENVIRONMENT AND FILES #####
    organizer = organize_experiment_info(run_file_storage_path, setup_file)
    split_ids = {'train': sample_split_ids, 'val': sample_split_ids, 'test': test_ids}

    ##### SETUP DATASET #####
    data_module = setup_data_module(SONData, exp_params, data_container, transforms, split_ids=split_ids, single_batch=True)

    ##### SETUP MODEL #####
    net = setup_model(backbone, exp_params, use_embeddings=True, model_type="learned_prompt_context")
    if state:
        net.load_state_dict(state, strict=False)
    model = CLIPFewShotModule(organizer.root_path+'outputs/', net, run_name, label_info, ['train', 'val', 'test'], VarTrackerRegression)
    model.set_hyperparams(lr, exp_params['hyperparams']['optim']['decay'])

    ##### SETUP TRAINER #####
    trainer = setup_trainer(organizer, run_name, exp_params, to_monitor="val_r2", monitor_mode='max')
 
    ##### FIT MODEL #####
    trainer.fit(model, datamodule=data_module)

###################
# TEST BEST MODEL #
###################

    for i, param in enumerate(model.parameters()):
        param.requires_grad_(False)

    top_model_path = get_top_model(run_file_storage_path, min_epoch)

    state = torch.load(top_model_path)['state']
    model.load_state_dict(state, strict=False)    

    tb_logger = TensorBoardLogger(save_dir=organizer.logs_path, name=run_name)
    tester = Trainer(gpus=exp_params['hyperparams']['gpu_nums'], logger=tb_logger)
    tester.test(model, datamodule=data_module)

# import pickle
# debug_ids = load_csv(exp_params['paths']['splits_root']+f'500.csv')[0]
# debug_labels = data_container.labels[data_container.labels['ID'].isin(debug_ids)]
# debug_labels.to_file("data/son_debug_500.geojson", driver="GeoJSON")

# debug_ids = [str(i) for i in debug_ids]
# debug_embeddings = { k:v for (k,v) in data_container.embeddings.items() if k in debug_ids}
# with open(f"data/embeddings/ViT-L-14_son_debug_500.pkl", 'wb') as f:
#     pickle.dump(debug_embeddings, f)