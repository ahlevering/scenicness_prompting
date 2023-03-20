import yaml
from pathlib import Path
import pickle

import clip
import torch
import numpy as np
import pandas as pd
from pytorch_lightning.utilities.seed import seed_everything
from sklearn.linear_model import Ridge

from scipy.stats import kendalltau, linregress
from sklearn.metrics import mean_squared_error

from codebase.pt_funcs.dataloaders import SONData, ClipDataLoader, SoNDataContainer, load_pickle
from codebase.pt_funcs.models_few_shot import ContrastiveManyPromptsNet, CLIPFewShotModule
from codebase.utils.file_utils import load_csv, make_crossval_splits

from codebase.experiment_tracking.save_metadata import ExperimentOrganizer
    
##### SET GLOBAL OPTIONS ######
seed_everything(113)
np.set_printoptions(suppress=True)
torch.set_printoptions(sci_mode=False)

##### LOAD SET-UP FILE #####
setup_file = "setup_files/train/son_few_shot_regression.yaml"
with open(setup_file) as file:
    exp_params = yaml.full_load(file)

run_name = exp_params['descriptions']['name']
run_family = exp_params['descriptions']['exp_family']

# Tokenize prompts
prompts = list(exp_params['hyperparams']['coop']['prompts'].keys())
prompts = torch.cat([clip.tokenize(p+".") for p in prompts])
prompts = prompts.to(device=exp_params['hyperparams']['gpu_nums'][0])

# Make weights learnable
architecture = "ViT-L/14"
archi_save_name = architecture.replace("/", "-") # Why did they use slashes in their naming!?

embeddings_path = exp_params['paths']['embeddings_root']+f"{archi_save_name}.pkl"
embeddings = load_pickle(embeddings_path)
# prompt_weights = torch.tensor(list(exp_params['hyperparams']['coop']['prompts'].values())).to(device=exp_params['hyperparams']['gpu_nums'][0])

data_container = SoNDataContainer(exp_params['paths']['labels_file'])
labels = data_container.labels

net, preprocess = clip.load(architecture)
net = net.to(device=exp_params['hyperparams']['gpu_nums'][0])
feature_extractor = ContrastiveManyPromptsNet(net, prompts, use_embeddings=True)

k_folds = exp_params['hyperparams']['k_folds']
best_performances = {}
columns = ["n_samples", "alpha", "rmse", "rsquared", "tau"]
for v in columns:
    best_performances[v] = []

## Forward pass using image & text features
i = 0
all_activations = {}
with torch.no_grad():
    while i <= len(data_container.labels):
        max_batch = min(128, len(data_container.labels) - i)
        ids = data_container.labels.iloc[i:i+max_batch]['ID']
        batch = torch.stack([embeddings[str(id_num)] for id_num in ids]).to(device=exp_params['hyperparams']['gpu_nums'][0])
        batch_activations = feature_extractor(batch).detach().cpu().numpy().squeeze()
        for index, id in enumerate(ids):
            all_activations[str(id)] = batch_activations[index]
        i += 128

with torch.no_grad():
    for n_samples in [25, 50, 100, 250, 500]:
        alpha_rsquared = []        
        for alpha in exp_params['hyperparams']['alpha']:
            all_split_ids = load_csv(exp_params['paths']['splits_root']+f'{n_samples}.csv')[0]
            all_split_ids = [int(r) for r in all_split_ids]
            train_ids, val_ids = make_crossval_splits(all_split_ids, k_folds)
            k_rsquared = []
            for k in range(k_folds):
                split_ids = {'train': train_ids[k], 'val': val_ids[k]}
                base_path = f"runs/{run_family}/{run_name}/{n_samples}/{alpha}/split_{k}/"
                organizer = ExperimentOrganizer(base_path)  

                train_gt = labels[labels['ID'].isin(split_ids['train'])]['Average'].values
                val_gt = labels[labels['ID'].isin(split_ids['val'])]['Average'].values
                train_activations = np.stack([all_activations[str(id_num)] for id_num in split_ids['train']])
                # train_embeddings = [embeddings[str(k)].to(device=exp_params['hyperparams']['gpu_nums'][0]) for k in split_ids['train']]
                # train_activations = np.stack([feature_extractor.forward(s).detach().cpu().numpy().squeeze() for s in train_embeddings])
                
                rr_model = Ridge(alpha=alpha)
                rr_model = Ridge(alpha=0.9)
                rr_model = rr_model.fit(train_activations, train_gt)
                # tes = rr_model.fit(train_gt.reshape(-1,1), train_gt)

                val_activations = np.stack([all_activations[str(id_num)] for id_num in split_ids['val']])
                # val_embeddings = [embeddings[str(k)].to(device=exp_params['hyperparams']['gpu_nums'][0]) for k in split_ids['val']]
                # val_activations = np.stack([feature_extractor.forward(s).detach().cpu().numpy().squeeze() for s in val_embeddings])
                val_preds = rr_model.predict(val_activations)
                # val_preds = rr_model.predict(val_gt.reshape(-1,1))

                _, _, rsquared, _, _ = linregress(val_preds, val_gt)
                # with open(str(organizer.states_path)+f"{rsquared}.pkl", 'wb') as f:
                #     pickle.dump(rr_model, f)

                k_rsquared.append(rsquared)
            # Aggregate metric over all k splits
            alpha_rsquared.append(np.mean(k_rsquared))

        ##### Test Model #####
        test_ids = data_container.labels[~data_container.labels['ID'].isin(all_split_ids)]
        test_ids = list(test_ids['ID'].values)
        test_split_gt = labels[labels['ID'].isin(test_ids)]['Average'].values

        ##### STORE ENVIRONMENT AND FILES #####
        base_path = f"runs/{run_family}/{run_name}/{n_samples}/best/{alpha}"
        organizer = ExperimentOrganizer(base_path)
        organizer.store_yaml(setup_file)
        organizer.store_environment()
        organizer.store_codebase(['.py'])

        # Get hyperparam performance
        best_index = np.argmax(alpha_rsquared)
        best_alpha = exp_params['hyperparams']['alpha'][best_index]

        all_split_activations = np.stack([all_activations[str(id_num)] for id_num in all_split_ids])
        all_split_gt = labels[labels['ID'].isin(all_split_ids)]['Average']

        rr_model = Ridge(alpha=best_alpha)
        rr_model = rr_model.fit(all_split_activations, all_split_gt)

        # test_split_embeddings = [embeddings[str(k)].to(device=exp_params['hyperparams']['gpu_nums'][0]) for k in test_ids]
        # test_split_activations = np.stack([feature_extractor.forward(s).detach().cpu().numpy().squeeze() for s in test_split_embeddings])        
        test_split_activations = np.stack([all_activations[str(id_num)] for id_num in test_ids])
        test_split_preds = rr_model.predict(test_split_activations)

        _, _, rsquared, _, _ = linregress(test_split_preds, test_split_gt)
        rmse = round(mean_squared_error(test_split_preds, test_split_gt, squared=False), 4)
        tau_corr = round(mean_squared_error(test_split_preds, test_split_gt, squared=False), 4)
        tau = round(kendalltau(test_split_preds, test_split_gt).correlation)

        best_performances["n_samples"].append(n_samples)
        best_performances["alpha"].append(best_alpha)
        best_performances["rmse"].append(rmse)
        best_performances["rsquared"].append(rsquared)
        best_performances["tau"].append(tau)
        # best_performances[f'{n_samples}'] = {'rmse': rmse, 'rsquared': rsquared, 'tau_corr':tau_corr}

        with open(str(organizer.states_path)+f"{rsquared}.pkl", 'wb') as f:
            pickle.dump(rr_model, f)

best_performances_df = pd.DataFrame(best_performances)
best_performances_df.to_csv(f"runs/{run_family}/{run_name}/metrics.csv", columns=columns)