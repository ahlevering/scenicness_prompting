import pickle as pkl
import geopandas as gpd
import pandas as pd
import torch
from transformers import AutoProcessor, AutoModel
from scipy.stats import kendalltau, linregress
from sklearn.metrics import mean_squared_error
from pathlib import Path

# Read the data files
# corine_pts = gpd.read_file("data/son_pts_aux_info.geojson")
son_pts = gpd.read_file("data/son_pts_with_bins.geojson")
son_pts.set_index('ID', inplace=True)    

encoder = "siglip"
with open(f"data/embeddings/embeds_{encoder}.pkl", 'rb') as f:
    embeddings = pkl.load(f)

data_path = "data/predictions_per_user.pkl"
all_prompts = pd.read_csv("data/prompts_cleaned.csv")
grouped_prompts = all_prompts.groupby('user_index')
prompts_in_order = list(all_prompts['prompts_in_order'].values)
values_in_order = list(all_prompts['values_in_order'].values)


embeds = [v for k, v in embeddings.items() if int(k) in son_pts.index] 
embeds_keys = [k for k in embeddings.keys() if int(k) in son_pts.index]

if encoder == "siglip":
    model = AutoModel.from_pretrained("google/siglip-base-patch16-224")#.to("cuda:0")
    processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224", do_rescale=True)
elif encoder == "clip":
    model = AutoModel.from_pretrained("openai/clip-vit-large-patch14")#.to("cuda:0")
    processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14", do_rescale=True)    
model.requires_grad_ = False

# Initialize a dictionary to store the confidence values for each prompt for each LC land cover value
emb_index = 0

early_ensembling_preds = []
user_preds = {}
with torch.no_grad():
    while len(embeds) > emb_index:
        upper_index = min(emb_index + 20000, len(embeds))
        batch_embeddings = embeds[emb_index:upper_index]
        
        image_embeds = torch.stack(batch_embeddings)

        if encoder == "siglip":
            inputs = processor(text=prompts_in_order, images=None, padding="max_length", return_tensors="pt")#.to("cuda:0")
            text_embeds = model.text_model(inputs['input_ids'])['pooler_output']        
            text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)    
            logits_per_text = torch.matmul(text_embeds, image_embeds.T) * model.logit_scale.exp() + model.logit_bias        
        elif encoder == "clip":
            inputs = processor(text=prompts_in_order, images=None, padding=True, return_tensors="pt")#.to("cuda:0")
            text_embeds = model.text_model(inputs['input_ids'])['pooler_output']        
            text_embeds = model.text_projection(text_embeds)
            text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)    
            logits_per_text = torch.matmul(text_embeds, image_embeds.T) * model.logit_scale.exp()

        ###
        '''LATE ENSEMBLING'''
        ###

        activations = torch.softmax(logits_per_text, dim=0)
        early_preds = torch.sum(activations * torch.tensor(values_in_order).unsqueeze(1), dim=0)
        early_ensembling_preds.append(early_preds)

        # Predict for all raters individually
        for rater_id, group in grouped_prompts:
            group_indices = list(group.index.values)  # Indices of the prompts for each user

            logits_per_group = logits_per_text[group_indices, :]  # Select the logits for the current group
            group_values = torch.tensor(list(group['values_in_order'].values))  # Values for the current group
            activations_per_group = torch.softmax(logits_per_group, dim=0)
            batch_user_preds = torch.sum(activations_per_group * group_values.unsqueeze(1), dim=0)
            
            # Store the predictions for each user
            if rater_id not in user_preds:
                user_preds[rater_id] = batch_user_preds
            else:
                user_preds[rater_id] = torch.cat((user_preds[rater_id], batch_user_preds))
        emb_index += 20000

## CALCULATE METRICS ##
metrics_dict = {}  # Dictionary to store metrics
gt = list(son_pts['Average'].values)
early_ensembling_preds = list(torch.cat(early_ensembling_preds))

_, _, r_value, _, _ = linregress(early_ensembling_preds, gt)
r_value = round(r_value, 3)
rmse = round(mean_squared_error(torch.stack(early_ensembling_preds), gt, squared=False), 3)
tau = kendalltau(torch.stack(early_ensembling_preds), gt)
tau_corr = round(tau.correlation, 3)

metrics_dict['early'] = {
    'r_value': r_value,
    'rmse': rmse,
    'tau_corr': tau_corr
}

###
'''LATE ENSEMBLING'''
###

min_prompts = [2,5,8,10]
aggregated_preds = {}
late_ensembling_preds = son_pts[['Average', 'geometry']] 

for min_prompt in min_prompts:
    filtered_user_preds = {user: user_preds[user] for user, group in grouped_prompts if len(group) >= min_prompt and user in user_preds}
    preds = sum(filtered_user_preds.values()) / len(filtered_user_preds)
    late_ensembling_preds[f'preds_{min_prompt}'] = preds

    _, _, r_value, _, _ = linregress(preds, gt)
    r_value = round(r_value, 3)
    rmse = round(mean_squared_error(preds, gt, squared=False), 3)
    tau = kendalltau(preds, gt)
    tau_corr = round(tau.correlation, 3)

    # Save the metrics into the dictionary
    metrics_dict[min_prompt] = {
        'r_value': r_value,
        'rmse': rmse,
        'tau_corr': tau_corr
    }

## WRITE AS OUTPUTS ##
metrics_dict = pd.DataFrame(metrics_dict)
Path("data/outputs/ensembling/").mkdir(exist_ok=True, parents=True)
metrics_dict.to_csv(f"data/outputs/ensembling/ensembling_results_{encoder}.csv")
late_ensembling_preds.to_file(f"data/outputs/ensembling/ensembling_preds_{encoder}.gpkg", driver="GPKG")