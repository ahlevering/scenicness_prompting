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

from codebase.pt_funcs.dataloaders import SONData, ClipDataLoader
from codebase.pt_funcs.models_zero_shot import CLIPZeroShotModel, ContrastiveManyPromptsNet
from codebase.utils.file_utils import load_csv, make_crossval_splits

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

##### SET UP TRANSFORMS #####
transforms = {}
transforms['train'] = process_yaml.setup_transforms(exp_params['transforms']['train'])
transforms['val'] = process_yaml.setup_transforms(exp_params['transforms']['val'])
transforms['test'] = process_yaml.setup_transforms(exp_params['transforms']['test'])

all_data = ["A photo of a natural area.", 6,
                        "A photo of snow capped mountains", 10,
                        "A photo of a scenic lake.", 9,
                        "A photo of a scenic and snowy wonderland", 10,
                        "A photo of an idilic landscape", 9,
                        "A carpet of flowers in the forest", 9,
                        "A photo of a field.", 6,
                        "A photo of an unremarkable rural area.", 5,
                        "A photo of an urban area.", 4,
                        "A photo of a highway.", 1,
                        "A photo of a construction area.", 1,
                        "A photo of vehicles.", 1,
                        "Man made metal structures.", 1,
                        "An urban park", 5,
                        "Agricultural machinery", 2,
                        "A road between fields", 3]


contrastive_prompts = all_data[0:-2:2]
promps_values = all_data[1:-1:2]

prompts = torch.cat([clip.tokenize(p) for p in contrastive_prompts])
prompts = prompts.to(device=exp_params['hyperparams']['gpu_nums'][0])


promps_values = torch.tensor(promps_values).to(device=exp_params['hyperparams']['gpu_nums'][0])

k_folds = exp_params['hyperparams']['k_folds']
for n_samples in [25, 50, 75, 100, 175, 250, 350, 500]:
    # Load split indices
    split_indices = load_csv(exp_params['paths']['splits_root']+f'{n_samples}.csv')
    # Subdivide sampled indices into k-fold bins
    train_indices, val_indices = make_crossval_splits(split_indices, k_folds)
    for k in range(k_folds):
        # Set bin for k
        split_indices = {'train': train_indices[k], 'val': val_indices[k]}

        label_info = {}
        label_info['scenic'] = {}
        label_info['scenic']['ylims'] = [1, 10]

        ##### SETUP LOADERS #####
        data_module = ClipDataLoader( exp_params['hyperparams']['workers'],    
                                    exp_params['hyperparams']['batch_size'],
                                    data_class=SONData
                                    )

        data_module.setup_data_classes( exp_params['paths']['labels_file'],
                                        exp_params['paths']['images_root'],
                                        split_indices,
                                        embeddings=exp_params['paths']['embeddings'],
                                        transforms=transforms,
                                        id_col=exp_params['descriptions']['id_col'],
                                        splits=exp_params['descriptions']['splits']
                                    )

        loader_emulator = next(iter(data_module.train_data))

        ##### STORE ENVIRONMENT AND FILES #####
        base_path = f"runs/{run_family}/{run_name}/{n_samples}/train/{k}/"
        organizer = ExperimentOrganizer(base_path)
        organizer.store_yaml(setup_file)
        organizer.store_environment()
        organizer.store_codebase(['.py'])

        ##### SETUP MODEL #####
        # Freeze feature extractors
        for i, param in enumerate(model.parameters()):
            param.requires_grad_(False)
        
        net, preprocess = clip.load('ViT-B/32')
        net = ContrastiveManyPromptsNet(net, prompts,promps_values)
        model = CLIPZeroShotModel(organizer.root_path+'outputs/', net, run_name, label_info, ['test'])


        checkpoint_callback = ModelCheckpoint(
            monitor= 'val_mse',
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
                            # checkpoint_callback=False,
                            fast_dev_run=False,
                            # limit_train_batches=250,
                            # limit_val_batches=10,
                            # gradient_clip_val=1
                        )

        ##### FIT MODEL #####
        print(f"fitting {run_name}")
        trainer.fit(model, datamodule=data_module)
        
        ##### TEST MODEL #####
        data_module = ClipDataLoader(   16,    
                                        64,
                                        data_class=SONData
                                    )

        data_module.setup_data_classes( exp_params['paths']['splits_file'],
                                        exp_params['paths']['images_root'],
                                        {'train': f"data/splits/res_4/{n_samples}.csv"},
                                        exp_params['descriptions']['id_col'],
                                        exp_params['descriptions']['splits'],
                                        train_transforms=train_trans,
                                        test_transforms=test_trans,
                                    )

        for i, param in enumerate(model.parameters()):
            param.requires_grad_(False)

        best_model = next(Path(f"runs/{run_family}/{run_name}/{n_samples}/train/outputs/states/").glob('**/*'))
        state = torch.load(best_model)['prompter']
        model.load_state_dict(state, strict=False)    
        
        tester = Trainer(gpus=exp_params['hyperparams']['gpu_nums'], logger=tb_logger)
        tester.test(model, datamodule=data_module) 