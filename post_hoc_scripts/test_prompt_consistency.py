import pickle as pkl
import geopandas as gpd
import pandas as pd
import torch
from transformers import AutoProcessor, AutoModel
from scipy import stats

# Read the data files
corine_pts = gpd.read_file("data/son_pts_aux_info.geojson")

encoder = "clip"
with open(f"data/embeddings/embeds_{encoder}.pkl", 'rb') as f:
    embeddings = pkl.load(f)

# Keep relevant columns and rename them
corine_pts = corine_pts[["ID", "Average", "lc"]]
corine_pts["lc"] = corine_pts["lc"].str[:2]

corine_lc_values = corine_pts["lc"].values
corine_pts.set_index('ID', inplace=True)

data_path = "data/predictions_per_user.pkl"
all_prompts = pd.read_csv("data/prompts_cleaned.csv")
prompts_in_order = list(all_prompts['prompts_in_order'].values)

embeds = [v for v in embeddings.values()]
embeds_keys = [k for k in embeddings.keys()]

if encoder == "siglip":
    model = AutoModel.from_pretrained("google/siglip-base-patch16-224")#.to("cuda:0")
    processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224", do_rescale=True)
elif encoder == "clip":
    model = AutoModel.from_pretrained("openai/clip-vit-large-patch14")#.to("cuda:0")
    processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14", do_rescale=True)    
model.requires_grad_ = False

# Initialize a dictionary to store the confidence values for each prompt for each LC land cover value
emb_index = 0
lc_prompts = {}

while len(embeds) > emb_index:
    upper_index = min(emb_index + 20000, len(embeds))
    batch_embeddings = embeds[emb_index:upper_index]
    
    image_embeds = torch.stack(batch_embeddings)

    if encoder == "siglip":
        inputs = processor(text=prompts_in_order, images=None, padding="max_length", return_tensors="pt")#.to("cuda:0")
        text_embeds = model.text_model(inputs['input_ids'])['pooler_output']        
        logits_per_text = torch.matmul(text_embeds, image_embeds.T) * model.logit_scale.exp() + model.logit_bias
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)    
    elif encoder == "clip":
        inputs = processor(text=prompts_in_order, images=None, padding=True, return_tensors="pt")#.to("cuda:0")
        text_embeds = model.text_model(inputs['input_ids'])['pooler_output']        
        text_embeds = model.text_projection(text_embeds)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)    
        logits_per_text = torch.matmul(text_embeds, image_embeds.T) * model.logit_scale.exp()                          
    activations = torch.softmax(logits_per_text, dim=0)

    # For each key in embeds_keys, get the corresponding LC value and store the image-activated prompts
    for i, key in enumerate(embeds_keys[emb_index:upper_index]):
        if int(key) in corine_pts.index:
            lc_value = corine_pts.loc[int(key), 'lc']
            if lc_value not in lc_prompts:
                lc_prompts[lc_value] = {}
            # Get the confidence values for the current batch of embeddings
            conf_values = activations[:, i].tolist()
            # For each prompt and its corresponding confidence value, append the confidence value to the list for the prompt
            for prompt, conf in zip(prompts_in_order, conf_values):
                if prompt not in lc_prompts[lc_value]:
                    lc_prompts[lc_value][prompt] = []
                lc_prompts[lc_value][prompt].append(conf)

    emb_index = upper_index

# After all the embeddings have been processed, calculate the average confidence for each prompt for each LC value
for lc_value in lc_prompts:
    for prompt in lc_prompts[lc_value]:
        lc_prompts[lc_value][prompt] = sum(lc_prompts[lc_value][prompt]) / len(lc_prompts[lc_value][prompt])

# Now, for each LC value, sort the prompts based on their average confidence values
for lc_value in lc_prompts:
    lc_prompts[lc_value] = [(prompt, round(avg_conf, 3)) for prompt, avg_conf in lc_prompts[lc_value][:3]]

# Convert the LC classes to float values and sort them
sorted_lc_values = sorted(lc_prompts.keys(), key=lambda x: float(x) / 10)

# Initialize the LaTeX table
latex_table = "\\begin{tabular}{|c|p{5cm}|}\n"
latex_table += "\\hline\n"
latex_table += "LC Class & \\small Most Activated Prompts \\\\\n"
latex_table += "\\hline\n"

# For each LC value in the sorted list, add a row to the LaTeX table
for lc_value in sorted_lc_values:
    latex_table += f"{lc_value} & \\small ({lc_prompts[lc_value][0][1]}) {lc_prompts[lc_value][0][0]} \\\\ & \\small ({lc_prompts[lc_value][1][1]}) {lc_prompts[lc_value][1][0]} \\\\ & \\small ({lc_prompts[lc_value][2][1]}) {lc_prompts[lc_value][2][0]} \\\\\n"
    latex_table += "\\hline\n"

# End the LaTeX table
latex_table += "\\end{tabular}"

print(latex_table)