import yaml
from pathlib import Path

import clip
import torch
import numpy as np
from torch import nn
from torchvision.models import resnet50
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.seed import seed_everything
from codebase.utils.file_utils import load_csv

from codebase.pt_funcs.dataloaders import SONData, ClipDataLoader, SoNDataContainer
from codebase.pt_funcs.models_few_shot import SONCLIPFewShotNet, CLIPFewShotModule, CLIPLinearProbe, Baseline

from codebase.experiment_tracking import process_yaml
from codebase.experiment_tracking.save_metadata import ExperimentOrganizer
    
##### SET GLOBAL OPTIONS ######
seed_everything(113)
np.set_printoptions(suppress=True)
torch.set_printoptions(sci_mode=False)

##### LOAD SET-UP FILE #####
setup_file = "setup_files/train/son_baseline.yaml"
with open(setup_file) as file:
    exp_params = yaml.full_load(file)

run_name = exp_params['descriptions']['name']
run_family = exp_params['descriptions']['exp_family']

if not 'baseline' in run_name:
    model, preprocess = clip.load('ViT-B/32')
else:
    model = resnet50(pretrained=True)
    model.fc = nn.Linear(2048, 1)
    model.fc.bias.data.fill_(5)

##### SET UP TRANSFORMS #####
transforms = {}
transforms['train'] = process_yaml.setup_transforms(exp_params['transforms']['train'])
transforms['val'] = process_yaml.setup_transforms(exp_params['transforms']['val'])
transforms['test'] = process_yaml.setup_transforms(exp_params['transforms']['val'])

label_info = {}
label_info['scenic'] = {}
label_info['scenic']['ylims'] = [1, 10]

##### SETUP LOADERS #####
data_container = SoNDataContainer(exp_params['paths']['labels_file'])

train_ids = load_csv("data/splits/baseline_train.csv")
val_ids = load_csv("data/splits/baseline_val.csv")

trainval_ids = train_ids
trainval_ids.extend(val_ids)
test_ids = data_container.labels[~data_container.labels['ID'].isin(trainval_ids)]
test_ids = list(test_ids['ID'].values)

split_indices = {'train': train_ids, 'val': val_ids, 'test': test_ids}


data_module = ClipDataLoader( exp_params['hyperparams']['workers'],    
                            exp_params['hyperparams']['batch_size'],
                            data_class=SONData
                            )

data_module.setup_data_classes( data_container,
                                exp_params['paths']['images_root'],
                                split_indices,
                                embeddings=None, # exp_params['paths']['embeddings'],
                                transforms=transforms,
                                id_col=exp_params['descriptions']['id_col'],
                                splits=exp_params['descriptions']['splits']
                            )

loader_emulator = next(iter(data_module.train_data))

##### STORE ENVIRONMENT AND FILES #####
base_path = f"runs/{run_family}/{run_name}/21210/train/"
organizer = ExperimentOrganizer(base_path)
organizer.store_yaml(setup_file)
organizer.store_environment()
organizer.store_codebase(['.py'])

##### SETUP MODEL #####
# Freeze feature extractors
if not 'baseline' in run_name:
    for i, param in enumerate(model.parameters()):
        param.requires_grad_(False)

if 'linear_probe' in run_name:
    net = CLIPLinearProbe(model)
elif 'baseline' in run_name:
    net = Baseline(model)
else:    
    net = SONCLIPFewShotNet(model, exp_params['hyperparams']['coop'])

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
                    # check_val_every_n_epoch=999,
                    # #exp_params['hyperparams']['check_val_every_n'],
                    callbacks=[checkpoint_callback],
                    logger = tb_logger,
                    fast_dev_run=False,
                    # limit_train_batches=10,
                    # limit_val_batches=10,
                )

##### FIT MODEL #####
print(f"fitting {run_name}")
trainer.fit(model, datamodule=data_module)

# ##### TEST MODEL #####
data_module = ClipDataLoader(   8,    
                                64,
                                data_class=SONData
                            )

data_module.setup_data_classes( data_container,
                                exp_params['paths']['images_root'],
                                split_indices,
                                embeddings=None, # exp_params['paths']['embeddings'],
                                transforms=transforms,
                                id_col=exp_params['descriptions']['id_col'],
                                splits=['test']
                            )

for i, param in enumerate(model.parameters()):
    param.requires_grad_(False)

best_model = next(Path(f"runs/{run_family}/{run_name}/21210/train/outputs/states/").glob('**/*'))
state = torch.load(best_model)['prompter']
model.load_state_dict(state, strict=False)    

tester = Trainer(gpus=exp_params['hyperparams']['gpu_nums'], logger=tb_logger)
tester.test(model, datamodule=data_module) 