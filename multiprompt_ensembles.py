import yaml
import pickle
from pathlib import Path

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
setup_file = "setup_files/test/son_zero_shot_multiprompt.yaml"
# setup_file = "setup_files/test/son_coop_contrastive_zero_shot.yaml"

with open(setup_file) as file:
    exp_params = yaml.full_load(file)
    
labels, embeddings = get_label_embeds_paths(exp_params)
data_container = SoNDataContainer(labels, embeddings)

data_path = "data/predictions_per_user.pkl"
all_prompts = pd.read_csv("data/prompts_cleaned.csv")
grouped_prompts = all_prompts.groupby(["user_index"])

if not Path(data_path).exists():
    backbone = "ViT-L/14"
    run_name = exp_params['descriptions']['name']
    run_family = exp_params['descriptions']['exp_family']

    label_info = setup_label_info(["scenic"])

    ##### SETUP MODEL #####
    ##### Encode prompts #####
    transforms = {'test': None}

    ### RUN OVER ALL IMAGES FOR EACH SET OF PROMPTS ###
    prediction_sets = []
    with torch.no_grad():
        ##### STORE ENVIRONMENT AND FILES #####
        base_path = f"runs/{run_family}/{run_name}/"
        organizer = organize_experiment_info(base_path, setup_file)

        data_module = setup_data_module(SONData, exp_params, data_container, transforms, split_ids=None, single_batch=False)

        preds = []
        for user_id, prompt_set in grouped_prompts:
            if len(prompt_set) > 1:
                prompt_value_pairs = {prompt_set['prompts_in_order'].iloc[i]:prompt_set['values_in_order'].iloc[i] for i in range(len(prompt_set))}
                exp_params['hyperparams']['prompts'] = prompt_value_pairs# prompt_set['prompts_in_order'].values()

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
                preds.append(model.test_tracker.variables['scenic'].attrs['preds'])
                model.test_tracker.reset_epoch_vars()

    valid_ids = [p['user_index'].values.tolist() for _, p in grouped_prompts if len(p) > 1]
    valid_ids = [p[0] for p in valid_ids]
    user_predictions = {}
    for i, idn in enumerate(valid_ids):
        user_predictions[idn] = preds[i]

    with open(data_path, 'wb') as f:
        pickle.dump(user_predictions, f)
else:
    with open(data_path, 'rb') as f:
        user_predictions = pickle.load(f)

from scipy.stats import kendalltau, linregress
from sklearn.metrics import mean_squared_error

for min_prompt_n in [1,2,5,8,10]:
    n_ids = [p['user_index'].values.tolist() for _, p in grouped_prompts if len(p) >= min_prompt_n]
    lens = [len(p) for p in n_ids]
    print("total prompts: ", np.sum(lens))
    print("median: ", np.median(lens))
    n_ids = [p[0] for p in n_ids]          
    user_preds_n = [user_predictions[idn] for idn in n_ids]
    preds_matrix = np.array(user_preds_n)

    out_df = data_container.labels
    out_df = out_df.drop(['index', 'folder_num', 'split', 'bin'], axis=1)

    ## Scenicness prediction maps
    print(f"#### {min_prompt_n} ####")
    matrix_mean = np.mean(preds_matrix, axis=0)
    out_df['preds'] = matrix_mean
    rmse = np.sqrt(mean_squared_error(out_df['preds'], out_df['Average']))
    print(rmse)
    _, _, r_value, _, _ = linregress(out_df['preds'], out_df['Average'])
    print(r_value)
    tau = kendalltau(out_df['preds'], out_df['Average'])
    print(tau)
    print(n_ids)

    ## Histogram of predicted values?
    matrix_var = np.var(preds_matrix, axis=0)
    out_df['var_pred'] = matrix_var
    out_df['var_diff'] = out_df['var_pred'] - out_df['Variance']
#     out_df.to_file(f"data/outputs/minimum_{min_prompt_n}.geojson", driver="GeoJSON")

    ## Collage of most divisive images (SON, and for each configuration)
    