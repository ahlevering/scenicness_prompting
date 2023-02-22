import yaml
from datetime import datetime
import warnings

import clip
import torch
import numpy as np
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.seed import seed_everything

from codebase.pt_funcs.dataloaders import SONData, ClipDataLoader
from codebase.pt_funcs.models_zero_shot import CLIPZeroShotModel, ContrastiveManyPromptsNet
from codebase.experiment_tracking.save_metadata import ExperimentOrganizer
    
##### SET GLOBAL OPTIONS ######
seed_everything(113)
np.set_printoptions(suppress=True)
torch.set_printoptions(sci_mode=False)

#### Shutting up annoying warnings ####
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

##### LOAD SET-UP FILE #####
setup_file = "setup_files/test/son_zero_shot.yaml"
with open(setup_file) as file:
    exp_params = yaml.full_load(file)

run_name = exp_params['descriptions']['name']
run_family = exp_params['descriptions']['exp_family']

net, preprocess = clip.load('ViT-B/32')
# model = model.to(device=exp_params['hyperparams']['gpu_num'])

##### SET UP TRANSFORMS #####
transforms = {'test': None}

label_info = {}
label_info['scenic'] = {}
label_info['scenic']['ylims'] = [1, 10]

##### SETUP LOADERS #####
data_module = ClipDataLoader( exp_params['hyperparams']['workers'],    
                              exp_params['hyperparams']['batch_size'],
                              data_class=SONData
                            )

data_module.setup_data_classes( splits_file=exp_params['paths']['splits_file'],
                                imgs_root=exp_params['paths']['images_root'],
                                sample_files=None, # No sampling from the dataset, run on entire
                                embeddings=exp_params['paths']['embeddings'],
                                transforms=transforms,
                                id_col=exp_params['descriptions']['id_col'],
                                splits=exp_params['descriptions']['splits']
                            )

loader_emulator = next(iter(data_module.test_data))

##### STORE ENVIRONMENT AND FILES #####
run_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
print(f"\nExperiment start: {run_time}\n")
base_path = f"runs/{run_family}/{run_name}/{run_time}/train/"
organizer = ExperimentOrganizer(base_path)
organizer.store_yaml(setup_file)
organizer.store_environment()
organizer.store_codebase(['.py'])

##### SETUP MODEL #####
contrastive_prompts = ["A photo of a natural area.",
                        "A photo of mountains",
                        "A photo of a scenic lake.",
                        "A photo of a field.",
                        "A photo of an unremarkable rural area.",
                        "A photo of an urban area.",
                        "A photo of a highway.",
                        "A photo of a construction area.",
                        "A photo of vehicles."]

promps_values = [1,1,1,0,0,-1,-1,-1,-1]
# contrastive_prompts = ["Photo of stuff.",
#                         "Photo of a dinosaur."]
prompts = torch.cat([clip.tokenize(p) for p in contrastive_prompts])
prompts = prompts.to(device=exp_params['hyperparams']['gpu_num'])


promps_values = torch.tensor(promps_values).to(device=exp_params['hyperparams']['gpu_num'])


net = ContrastiveManyPromptsNet(net, prompts,promps_values)
model = CLIPZeroShotModel(organizer.root_path+'outputs/', net, run_name, label_info, ['test'])

##### SETUP TESTER #####
tb_logger = TensorBoardLogger(save_dir=organizer.logs_path, name=run_name)
trainer = Trainer(  gpus=[exp_params['hyperparams']['gpu_num']],
                    logger = tb_logger,
                    fast_dev_run=False,
                )

##### FIT MODEL #####
print(f"fitting {run_name}")
trainer.test(model, datamodule=data_module)