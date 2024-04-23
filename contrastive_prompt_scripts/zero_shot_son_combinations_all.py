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

with open(setup_file) as file:
    exp_params = yaml.full_load(file)
    
# sample_split_ids = get_sample_split_ids(exp_params, 25, to_int=True)#[10:20]

labels, embeddings = get_label_embeds_paths(exp_params)

data_container = SoNDataContainer(labels, embeddings)
# data_container.labels = data_container.labels[data_container.labels['ID'].isin(sample_split_ids)]

run_name = exp_params['descriptions']['name']
run_family = exp_params['descriptions']['exp_family']

##### SETUP MODEL #####
##### Encode prompts #####

model = AutoModel.from_pretrained("google/siglip-base-patch16-224")#.to("cuda:0")
model.requires_grad_ = False

transforms = {'test': None}

# split_ids = {'test': sample_split_ids}
data_module = setup_data_module(SONData, exp_params, data_container, transforms, split_ids=None, single_batch=False)
test_loader = data_module.test_dataloader()
processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224", do_rescale=True)

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
                # url = 'https://scenicornot.datasciencelab.co.uk/img/00/25/12/251237_f026cac7.jpg'
                # image = Image.open(requests.get(url, stream=True).raw)                
                inputs = processor(text=prompts, images=None, padding="max_length", return_tensors="pt")#.to("cuda:0")
                text_embeds = model.text_model(inputs['input_ids'])['pooler_output']
                text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)    
                predictions = []
                ground_truths = []

                if not batch:
                    emb_index = 0
                    while len(embeds) > emb_index:
                        upper_index = min(emb_index + 20000, len(embeds))
                        batch_embeddings = embeds[emb_index:upper_index]
                        
                        image_embeds = torch.stack(batch_embeddings)
                        logits_per_text = torch.matmul(text_embeds, image_embeds.T) * model.logit_scale.exp() + model.logit_bias
                        most_likely_prompt = torch.softmax(logits_per_text, dim=0)
                        scenicness = (most_likely_prompt[0, :] * 9) + 1

                        predictions.append(scenicness.cpu().numpy())
                        ground_truths.append(gt_values[emb_index:upper_index])
                        emb_index += 20000
                else:
                    image_embeds = torch.stack(batch['img'])#.to("cuda:0")
                    logits_per_text = torch.matmul(text_embeds, image_embeds.T) * model.logit_scale.exp() + model.logit_bias
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
Path("data/outputs/contrastive/").mkdir(parents=True, exist_ok=True)
results_df.to_csv("data/outputs/contrastive/zero_shot_contrasts_all.csv", header=True)
# net, preprocess = clip.load(architecture)

# values = list(exp_params['hyperparams']['coop']['prompts'].values())
# values = torch.tensor(values, requires_grad=False).to(device=exp_params['hyperparams']['gpu_nums'][0])

# net = PP2ManyPrompts(net, prompts, values, use_embeddings=True, son_rescale=True)    

# prompts = list(exp_params['hyperparams']['coop']['prompts'].keys())
# prompts = torch.cat([clip.tokenize(p+".") for p in prompts])
# prompts = prompts.to(device=exp_params['hyperparams']['gpu_nums'][0])