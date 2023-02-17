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

from codebase.pt_funcs.dataloaders import SONData, ClipDataLoader
from codebase.pt_funcs.models_few_shot import SONCLIPFewShotNet, CLIPFewShotModule

from codebase.experiment_tracking import process_yaml
from codebase.experiment_tracking.save_metadata import ExperimentOrganizer
    
##### SET GLOBAL OPTIONS ######
seed_everything(113)
np.set_printoptions(suppress=True)
torch.set_printoptions(sci_mode=False)

##### LOAD SET-UP FILE #####
setup_file = "setup_files/train/son_few_shot.yaml"
with open(setup_file) as file:
    exp_params = yaml.full_load(file)

run_name = exp_params['descriptions']['name']
run_family = exp_params['descriptions']['exp_family']

net, preprocess = clip.load('ViT-B/32')
# model = model.to(device=exp_params['hyperparams']['gpu_num'])

##### SET UP TRANSFORMS #####
train_trans = process_yaml.setup_transforms(exp_params['transforms']['train'])

label_info = {}
label_info['scenic'] = {}
# label_info['score']['index'] = 0
label_info['scenic']['ylims'] = [1, 10]

##### SETUP LOADERS #####
data_module = ClipDataLoader( exp_params['hyperparams']['workers'],    
                              exp_params['hyperparams']['batch_size'],
                              data_class=SONData
                            )

data_module.setup_data_classes( exp_params['paths']['splits_file'],
                                exp_params['paths']['images_root'],
                                exp_params['descriptions']['id_col'],
                                exp_params['descriptions']['splits'],
                                train_transforms=train_trans,
                            )

loader_emulator = next(iter(data_module.train_data))

##### STORE ENVIRONMENT AND FILES #####
run_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
print(f"\nExperiment start: {run_time}\n")
base_path = f"runs/{run_family}/{run_name}/{run_time}/train/"
organizer = ExperimentOrganizer(base_path)
organizer.store_yaml(setup_file)
organizer.store_environment()
organizer.store_codebase(['.py'])

##### SETUP MODEL #####
# contrastive_prompts = ["Photo of an extremely beautiful region.",
#                         "Photo of an extremely ugly region."]

# prompts = torch.cat([clip.tokenize(p) for p in contrastive_prompts])
# prompts = prompts.to(device=exp_params['hyperparams']['gpu_num'])

# Freeze feature extractors
for i, param in enumerate(net.parameters()):
    param.requires_grad_(False)
net = SONCLIPFewShotNet(net, exp_params['hyperparams']['coop'])

model = CLIPFewShotModule(organizer.root_path+'outputs/', net, run_name, label_info, exp_params['descriptions']['splits'])
model.set_hyperparams(exp_params['hyperparams']['optim']['lr'], exp_params['hyperparams']['optim']['decay'])

checkpoint_callback = ModelCheckpoint(
    monitor= 'train_mse',
    dirpath= str(organizer.states_path),
    filename='{epoch:02d}-{mse:.2f}',
    save_top_k=1,
    mode='min')

##### SETUP TESTER #####
tb_logger = TensorBoardLogger(save_dir=organizer.logs_path, name=run_name)
trainer = Trainer(  max_epochs=exp_params['hyperparams']['epochs'],
                    gpus=exp_params['hyperparams']['gpu_nums'],
                    # check_val_every_n_epoch=999,#exp_params['hyperparams']['check_val_every_n'],
                    callbacks=[checkpoint_callback],
                    logger = tb_logger,
                    # checkpoint_callback=False,
                    fast_dev_run=False,
                    limit_train_batches=250,
                    # limit_val_batches=10,
                    # gradient_clip_val=1
                )

##### FIT MODEL #####
print(f"fitting {run_name}")
trainer.fit(model, datamodule=data_module)