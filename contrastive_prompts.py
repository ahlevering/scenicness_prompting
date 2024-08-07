import yaml
import copy


from PIL import Image
from pathlib import Path
import requests
import torch
import numpy as np
import pandas as pd
from codebase.pt_funcs.dataloaders import SoNDataContainer, SONData
from transformers import AutoProcessor, AutoModel

from codebase.exp_wrappers import *

from scipy.stats import kendalltau, linregress
from sklearn.metrics import mean_squared_error

##### SET GLOBAL OPTIONS ######
torch.manual_seed(113)
np.set_printoptions(suppress=True)
torch.set_printoptions(sci_mode=False)

##### LOAD SET-UP FILE #####
setup_file = "setup_files/test/contrastive_prompting.yaml"
CALIBRATE = False

with open(setup_file) as file:
    exp_params = yaml.full_load(file)
    
labels, embeddings = get_label_embeds_paths(exp_params)

data_container = SoNDataContainer(labels, embeddings)
if CALIBRATE:
    sample_split_ids = get_sample_split_ids(exp_params, 25, to_int=True)
    data_container.labels = data_container.labels[data_container.labels['ID'].isin(sample_split_ids)]

run_name = exp_params['descriptions']['name']
run_family = exp_params['descriptions']['exp_family']

##### SETUP MODEL #####
##### Encode prompts #####
encoder = exp_params['hyperparams']['embeddings'].lower()
if encoder == "siglip":
    model = AutoModel.from_pretrained("google/siglip-base-patch16-224")#.to("cuda:0")
    processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224", do_rescale=True)
elif encoder == "clip":
    model = AutoModel.from_pretrained("openai/clip-vit-large-patch14")#.to("cuda:0")
    processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14", do_rescale=True)    
model.requires_grad_ = False

transforms = {'test': None}

if CALIBRATE:
    split_ids = {'test': sample_split_ids}
else:
    split_ids = None
data_module = setup_data_module(SONData, exp_params, data_container, transforms, split_ids=split_ids, single_batch=False)
test_loader = data_module.test_dataloader()

metrics = {"context":       [],
           "positive":      [],
           "negative":      [],
           "rmse":          [],
           "r_squared":     [],
           "kendalls_tau":  []}    

with torch.no_grad():
    if len(test_loader) == 1:
        batch = next(iter(test_loader))
    else:
        batch = False
        embeds = [v for v in data_container.embeddings.values()]
        gt_values = data_container.labels['Average'].values
    for ctx in exp_params['hyperparams']['prompts']['contexts']:
        exp_params['hyperparams']['prompts']['ctx_init'] = ctx
        for pos in exp_params['hyperparams']['prompts']['positive']:
            for neg in exp_params['hyperparams']['prompts']['negative']:
                prompts = [f"{ctx} {pos}", f"{ctx} {neg}"]
                                
                if encoder == "siglip":
                    inputs = processor(text=prompts, images=None, padding="max_length", return_tensors="pt")#.to("cuda:0")
                    text_embeds = model.text_model(inputs['input_ids'])['pooler_output']   
                    text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
                elif encoder == "clip":
                    inputs = processor(text=prompts, images=None, padding=True, return_tensors="pt")#.to("cuda:0")
                    text_embeds = model.text_model(inputs['input_ids'])['pooler_output']        
                    text_embeds = model.text_projection(text_embeds)            
                    text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)              
                predictions = []
                ground_truths = []

                if not batch:
                    emb_index = 0
                    while len(embeds) > emb_index:
                        upper_index = min(emb_index + 20000, len(embeds))
                        batch_embeddings = embeds[emb_index:upper_index]
                        image_embeds = torch.stack(batch_embeddings)

                        if encoder == "siglip":                  
                            logits_per_text = torch.matmul(text_embeds, image_embeds.T) * model.logit_scale.exp() + model.logit_bias
                        elif encoder == "clip":
                            logits_per_text = torch.matmul(text_embeds, image_embeds.T) * model.logit_scale.exp()                

                        most_likely_prompt = torch.softmax(logits_per_text, dim=0)
                        scenicness = (most_likely_prompt[0, :] * 9) + 1

                        predictions.append(scenicness.cpu().numpy())
                        ground_truths.append(gt_values[emb_index:upper_index])
                        emb_index += 20000
                else:
                    # image_embeds = torch.stack(batch['img'])#.to("cuda:0")
                    if encoder == "siglip":                  
                        logits_per_text = torch.matmul(text_embeds, batch['img'].T) * model.logit_scale.exp() + model.logit_bias
                    elif encoder == "clip":
                        logits_per_text = torch.matmul(text_embeds, batch['img'].T) * model.logit_scale.exp()  
                    most_likely_prompt = torch.softmax(logits_per_text, dim=0)
                    scenicness = (most_likely_prompt[0, :] * 9) + 1

                    predictions.append(scenicness.cpu().numpy())
                    ground_truths.append(copy.deepcopy(batch['gt'].cpu().numpy()))                    


                predictions = np.concatenate(predictions)
                ground_truths = np.concatenate(ground_truths)
                
                _, _, r_value, _, _ = linregress(predictions, ground_truths)
                r_value = round(r_value, 3)
                rmse = round(mean_squared_error(predictions, ground_truths, squared=False), 3)
                tau = kendalltau(predictions, ground_truths)
                tau_corr = round(tau.correlation, 3)

                metrics["context"].append(ctx)
                metrics["positive"].append(pos)
                metrics["negative"].append(neg)

                metrics["rmse"].append(rmse)
                metrics["r_squared"].append(r_value)
                metrics["kendalls_tau"].append(tau_corr)                                

results_df = pd.DataFrame(metrics)
out_dir = f"data/outputs/contrastive/{encoder}/"
Path(out_dir).mkdir(parents=True, exist_ok=True)
if CALIBRATE:
    results_df.to_csv(f"{out_dir}zero_shot_contrasts_25.csv", header=True)
else:
    results_df.to_csv(f"{out_dir}zero_shot_contrasts.csv", header=True)