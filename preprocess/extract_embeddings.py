import yaml
import pickle
from pathlib import Path

import torch
import numpy as np
from PIL import Image
import requests
from transformers import AutoProcessor, AutoModel
import torch
from torchvision.transforms import ToTensor

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from codebase.pt_funcs.dataloaders import SONData, ClipDataLoader, SoNDataContainer
from codebase.experiment_tracking import process_yaml

##### SET GLOBAL OPTIONS ######
np.set_printoptions(suppress=True)
torch.set_printoptions(sci_mode=False)

setup_file = "setup_files/test/extract_embeddings.yaml"
with open(setup_file) as file:
    exp_params = yaml.full_load(file)
out_dir = "data/embeddings/"
Path(out_dir).mkdir(exist_ok=True, parents=True)

encoder = exp_params['hyperparams']['embeddings'].lower()
if encoder == "siglip":
    model = AutoModel.from_pretrained("google/siglip-base-patch16-224").to("cuda:0")
    processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224", do_rescale=True)
elif encoder == "clip":
    model = AutoModel.from_pretrained("openai/clip-vit-large-patch14").to("cuda:0")
    processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14", do_rescale=True)    
model.requires_grad_ = False

# url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
# image = Image.open(requests.get(url, stream=True).raw)
# inputs = processor(text=[""], images=image, padding="max_length", return_tensors="pt")

# inputs = processor(text=[""], images=batch['img'], padding="max_length", return_tensors="pt")

with torch.no_grad():
    data_container = SoNDataContainer(exp_params['paths']['labels_file'])

    ##### SETUP LOADERS #####
    data_module = ClipDataLoader(24, 256, data_class=SONData)
    data_module.setup_data_classes( data_container,
                                    exp_params['paths']['images_root'],
                                    None,
                                    embeddings_policy={'test': False}, # exp_params['paths']['embeddings'],
                                    transforms={'test': False},
                                    splits=['all']
                                )

    embeddings = {}
    with torch.no_grad():
        for batch in iter(data_module.test_dataloader()):
            # inputs = processor(text=None, images=batch['img'], return_tensors="pt").to("cuda:0")
            inputs = processor(text=["an image of a beautiful landscape", "an image of an ugly landscape"], images=batch['img'], padding=True, return_tensors="pt").to("cuda:0")
            image_embeds = model.vision_model(inputs.pixel_values).pooler_output
            if encoder == "clip":
                image_embeds = model.visual_projection(image_embeds)
            # text_embeds = model.text_model(inputs.input_ids).pooler_output
            image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
            # text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)            

            # logits_per_text = torch.matmul(text_embeds, image_embeds.T) * model.logit_scale.exp() + model.logit_bias
            # most_likely_prompt = torch.softmax(logits_per_text, dim=0)
            # logits_per_image = logits_per_text.t()

            for i, id_num in enumerate(batch['ids']):
                embeddings[str(int(id_num))] = image_embeds[i,:].detach().cpu()

        with open(f"{out_dir}embeds_{encoder}.pkl", 'wb') as f:
            pickle.dump(embeddings, f)