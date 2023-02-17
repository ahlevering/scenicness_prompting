import yaml
from datetime import datetime
import warnings

import clip
import torch
import numpy as np
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.seed import seed_everything

from codebase.pt_funcs.dataloaders import PP2Data, PP2DataContainer, ClipDataLoader
from codebase.pt_funcs.models_zero_shot import PP2CLIPNet, CLIPZeroShotModel, CLIPZeroShotPP2

from codebase.experiment_tracking import process_yaml
from codebase.experiment_tracking.save_metadata import ExperimentOrganizer
    
##### SET GLOBAL OPTIONS ######
seed_everything(113)
np.set_printoptions(suppress=True)
torch.set_printoptions(sci_mode=False)

#### Shutting up annoying warnings ####
# warnings.simplefilter(action='ignore', category=UserWarning)
# warnings.simplefilter(action='ignore', category=FutureWarning)

##### LOAD SET-UP FILE #####
setup_file = "setup_files/train/zero_shot_pp2.yaml"
with open(setup_file) as file:
    exp_params = yaml.full_load(file)

run_name = exp_params['descriptions']['name']
run_family = exp_params['descriptions']['exp_family']

model, preprocess = clip.load('ViT-B/32')
# model = model.to(device=exp_params['hyperparams']['gpu_num'])

##### SET UP TRANSFORMS #####
test_trans = preprocess # process_yaml.setup_transforms(exp_params['transforms']['train'])

labels = ['lively', 'depressing', 'boring', 'beautiful', 'safe', 'wealthy']
inverse = ['boring', 'inspiring', 'lively', 'ugly', 'unsafe', 'impoverished']

label_info = {}
for l in labels:
    label_info[l] = {}
    label_info[l]['ylims'] = [-2, 2]

##### SETUP LOADERS #####
data_module = ClipDataLoader( exp_params['hyperparams']['workers'],    
                            exp_params['hyperparams']['batch_size'],
                            data_class=PP2Data,
                            container_class=PP2DataContainer
                        )

data_module.setup_data_classes( exp_params['paths']['splits_file'],
                                exp_params['paths']['images_root'],
                                exp_params['descriptions']['id_col'],
                                exp_params['descriptions']['splits'],
                                test_transforms=test_trans,
                            )

# What do you mean I could just adjust the GT?
cols = [str.capitalize(c) for c in labels]
cols.remove("Safe")
cols.append("Safety")
data_module.exp_data.normalize(cols)                            

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
prompts = []
for i, l in enumerate(labels):
    l_prompts = [f"Photo of an extremely {l} area.",
                 f"Photo of an extremely {inverse[i]} area."]
    p = torch.cat([clip.tokenize(p) for p in l_prompts]).to(device=exp_params['hyperparams']['gpu_num'])
    prompts.append(p)

net = PP2CLIPNet(model, prompts)
model = CLIPZeroShotPP2(organizer.root_path+'outputs/', net, run_name, label_info, ['test'])

##### SETUP TESTER #####
tb_logger = TensorBoardLogger(save_dir=organizer.logs_path, name=run_name)
trainer = Trainer(  gpus=[exp_params['hyperparams']['gpu_num']],
                    logger = tb_logger,
                    fast_dev_run=False,
                )

##### FIT MODEL #####
print(f"fitting {run_name}")
trainer.test(model, datamodule=data_module)