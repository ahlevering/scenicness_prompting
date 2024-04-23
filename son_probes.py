import yaml
import pickle
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from codebase.pt_funcs.dataloaders import SoNDataContainer
from sklearn.linear_model import SGDRegressor

from codebase.exp_wrappers import *

from scipy.stats import kendalltau, linregress
from sklearn.metrics import mean_squared_error
    
##### SET GLOBAL OPTIONS ######
torch.manual_seed(113)
np.set_printoptions(suppress=True)
torch.set_printoptions(sci_mode=False)

##### LOAD SET-UP FILE #####
setup_file = "setup_files/test/contrastive_prompting.yaml"

with open(setup_file) as file:
    exp_params = yaml.full_load(file)
    
# sample_split_ids = get_sample_split_ids(exp_params, 25, to_int=True)#[10:20]

labels, embeddings = get_label_embeds_paths(exp_params)
k_folds = 5

learning_rates = [100, 50, 25, 10, 5, 1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.00025, 0.0001, 0.000075, 0.00005, 0.000025, 0.00001]

data_container = SoNDataContainer(labels, embeddings)
data_container.labels = data_container.labels.astype({'ID':str})
# data_container.labels = data_container.labels[data_container.labels['ID'].isin(sample_split_ids)]

run_name = exp_params['descriptions']['name']
run_family = exp_params['descriptions']['exp_family']

# For inference pass
image_embeds = [v for v in data_container.embeddings.values()]
gt_values = data_container.labels['Average'].values

output_predictions = gpd.GeoDataFrame({
    'id': data_container.labels['ID'].values,
    'lat': data_container.labels['Lat'].values,
    'lon': data_container.labels['Lon'].values,
    'GT': data_container.labels['Average'].values,
})

output_predictions['geom'] = list(zip(output_predictions.lon, output_predictions.lat))
output_predictions['geom'] = output_predictions['geom'].apply(Point)
output_predictions.set_geometry('geom', inplace=True)
output_predictions.set_crs(epsg=4326, inplace=True)

full_model_weights = {}
full_model_metrics = pd.DataFrame(columns=['n_samples', 'lr', 'rmse', 'r_squared', 'kendalls_tau'])

def calc_metrics(preds, gt):
    _, _, r_value, _, _ = linregress(preds, gt)
    r_value = round(r_value, 3)
    rmse = round(mean_squared_error(preds, gt, squared=False), 3)
    tau = kendalltau(preds, gt)
    tau_corr = round(tau.correlation, 3)
    return r_value, rmse, tau_corr

fold_model_metrics = pd.DataFrame(columns=['n_samples', 'lr', 'fold', 'rmse', 'r_squared', 'kendalls_tau'])
for n_samples in [25, 50, 100, 250, 500]:
    sample_split_ids = get_sample_split_ids(exp_params, n_samples, to_int=False)
    train_ids, val_ids, _ = get_crossval_splits(sample_split_ids, k_folds, data_container)

    ### TRAINING PASS ###
    for lr in learning_rates:        
        for k in range(k_folds):
            ### Fit model ###
            train_gt_fold_k = data_container.labels[data_container.labels['ID'].isin(train_ids[k])]
            train_embeds_fold_k = np.stack([data_container.embeddings[str(id_num)] for id_num in train_gt_fold_k['ID'].values])
 
            model = SGDRegressor(learning_rate='adaptive', loss='huber', eta0=lr)#, early_stopping=False, tol=1e-3)#, penalty=None)
            model.intercept_ = np.array([5.5])
            model.fit(train_embeds_fold_k, train_gt_fold_k['Average'].values)
            
            ### Predict over hold-out set ###
            val_gt_fold_k = data_container.labels[data_container.labels['ID'].isin(val_ids[k])]
            val_embeds_fold_k = np.stack([data_container.embeddings[str(id_num)] for id_num in val_gt_fold_k['ID'].values])
     
            preds = model.predict(val_embeds_fold_k)
            r_value, rmse, tau_corr = calc_metrics(preds, val_gt_fold_k['Average'].values)
            metrics_dict = {
                'n_samples': n_samples,
                'lr': lr,
                'fold': k,
                'rmse': rmse,
                'r_squared': r_value,
                'kendalls_tau': tau_corr
            }              
            fold_model_metrics = fold_model_metrics._append(metrics_dict, ignore_index=True)

    ### RETRIEVE BEST LR & LOAD COEFFICIENTS ###
    avg_r_squared = fold_model_metrics.groupby('lr')['r_squared'].mean()
    best_lr = avg_r_squared.idxmax()

    finetuned_model = SGDRegressor(learning_rate='adaptive', loss='huber', eta0=best_lr)
    finetuned_model.intercept_ = np.array([5.5])

    ### FIT MODEL ON ALL SAMPLES ###
    train_gt_full = data_container.labels[data_container.labels['ID'].isin(sample_split_ids)]
    train_embeds_full = np.stack([data_container.embeddings[str(id_num)] for id_num in train_gt_full['ID'].values])

    finetuned_model.fit(train_embeds_full, train_gt_full['Average'].values)
    full_model_weights[n_samples] = [finetuned_model.coef_, finetuned_model.intercept_]

    ### INFERENCE PASS ###
    predictions = []
    ground_truths = []

    emb_index = 0
    while len(image_embeds) > emb_index:
        upper_index = min(emb_index + 20000, len(image_embeds))
        batch_embeddings = image_embeds[emb_index:upper_index]
        
        batch_embeddings = np.stack(batch_embeddings)
        preds = finetuned_model.predict(batch_embeddings)

        predictions.append(preds)
        ground_truths.append(gt_values[emb_index:upper_index])
        emb_index += 20000
    r_value, rmse, tau_corr = calc_metrics(np.concatenate(predictions), np.concatenate(ground_truths))
    metrics_dict = {
        'n_samples': n_samples,
        'lr': lr,
        'rmse': rmse,
        'r_squared': r_value,
        'kendalls_tau': tau_corr
    }              
    full_model_metrics = full_model_metrics._append(metrics_dict, ignore_index=True)
    output_predictions[f'pred_{n_samples}'] = np.concatenate(predictions)

Path("data/outputs/").mkdir(parents=True, exist_ok=True)
full_model_metrics.to_csv("data/outputs/probe_results.csv")
output_predictions.to_file("data/outputs/probe_predictions.gpkg", driver="GPKG")

with open('data/outputs/probe_weights_dict.pkl', 'wb') as f:
    pickle.dump(full_model_weights, f)