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
from codebase.pt_funcs.models_few_shot import SONCLIPFewShotNet, CLIPFewShotModule, CLIPLinearProbe

from codebase.experiment_tracking import process_yaml
from codebase.experiment_tracking.save_metadata import ExperimentOrganizer
    
##### SET GLOBAL OPTIONS ######
seed_everything(113)
np.set_printoptions(suppress=True)
torch.set_printoptions(sci_mode=False)

##### LOAD SET-UP FILE #####
setup_file = "setup_files/test/son_linear_probe.yaml"
with open(setup_file) as file:
    exp_params = yaml.full_load(file)

run_name = exp_params['descriptions']['name']
run_family = exp_params['descriptions']['exp_family']
train_dir = f"{exp_params['paths']['train_dir']}"

clip_model, preprocess = clip.load('ViT-B/32')
# model = model.to(device=exp_params['hyperparams']['gpu_num'])

##### SET UP TRANSFORMS #####
test_trans = process_yaml.setup_transforms(exp_params['transforms']['test'])

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
                                None, #"data/splits/res_4/500.csv",
                                exp_params['descriptions']['id_col'],
                                exp_params['descriptions']['splits'],
                                test_transforms=test_trans,
                            )

loader_emulator = next(iter(data_module.test_data))

##### STORE ENVIRONMENT AND FILES #####
run_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
print(f"\nExperiment start: {run_time}\n")
base_path = train_dir+f'test/{run_time}/' # Add to existing dir
organizer = ExperimentOrganizer(base_path)
organizer.store_yaml(setup_file)
organizer.store_environment()
organizer.store_codebase(['.py'])

##### SETUP MODEL #####
if 'linear_probe' in run_name:
    net = CLIPLinearProbe(clip_model)
else:    
    net = SONCLIPFewShotNet(clip_model, exp_params['hyperparams']['coop'])

model = CLIPFewShotModule(organizer.root_path+'outputs/', net, run_name, label_info, exp_params['descriptions']['splits'])
if 'weights_file' in exp_params['paths']:
    prompt_state = torch.load(exp_params['paths']['weights_file'])['prompter']#, map_location=torch.device('cpu'))['prompt_state']
    model.load_state_dict(prompt_state, strict=False)
# model.set_hyperparams(exp_params['hyperparams']['optim']['lr'], exp_params['hyperparams']['optim']['decay'])

##### SETUP TESTER #####
tb_logger = TensorBoardLogger(save_dir=organizer.logs_path, name=run_name)
model = model.eval()

##### SETUP TRAINER #####
tb_logger = TensorBoardLogger(save_dir=organizer.logs_path)
tester = Trainer(gpus=exp_params['hyperparams']['gpu_nums'], logger=tb_logger)
##### FIT MODEL #####
tester.test(model, datamodule=data_module)