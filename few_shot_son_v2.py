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

from codebase.pt_funcs.dataloaders import SoNDataContainer, SONData, ClipDataLoader
from codebase.pt_funcs.models_few_shot import SONCLIPFewShotNet, CLIPFewShotModule, CLIPLinearProbe, Baseline
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

data_container = SoNDataContainer(exp_params['paths']['labels_file'])    

run_name = exp_params['descriptions']['name']
run_family = exp_params['descriptions']['exp_family']

##### SET UP TRANSFORMS #####
transforms = {}
transforms['train'] = process_yaml.setup_transforms(exp_params['transforms']['train'])
transforms['val'] = process_yaml.setup_transforms(exp_params['transforms']['val'])
transforms['test'] = process_yaml.setup_transforms(exp_params['transforms']['test'])

k_folds = exp_params['hyperparams']['k_folds']
for n_samples in [25, 50, 75, 100, 175, 250, 325, 400, 500]:
    # Load split indices
    split_ids = load_csv(exp_params['paths']['splits_root']+f'{n_samples}.csv')
    # Subdivide sampled indices into k-fold bins
    train_ids, val_ids = make_crossval_splits(split_ids, k_folds)
    trainval_ids = train_ids
    trainval_ids.extend(val_ids)
    test_ids = data_container.labels[~data_container.labels['ID'].isin(trainval_ids)]
    test_ids = list(test_ids['ID'].values)

    for k in range(k_folds):
        # Set bin for k
        split_ids = {'train': train_ids[k], 'val': val_ids[k], 'test': test_ids}
        if not 'baseline' in run_name:
            model, preprocess = clip.load('ViT-B/32')
        else:
            pass # TODO: figure this crap out later

        label_info = {}
        label_info['scenic'] = {}
        label_info['scenic']['ylims'] = [1, 10]

        ##### SETUP LOADERS #####
        data_module = ClipDataLoader( exp_params['hyperparams']['workers'],    
                                    exp_params['hyperparams']['batch_size'],
                                    data_class=SONData
                                    )

        data_module.setup_data_classes( data_container,
                                        exp_params['paths']['images_root'],
                                        split_ids,
                                        embeddings=None, # exp_params['paths']['embeddings'],
                                        transforms=transforms,
                                        id_col=exp_params['descriptions']['id_col'],
                                        splits=exp_params['descriptions']['splits']
                                    )

        loader_emulator = next(iter(data_module.train_data))

        ##### STORE ENVIRONMENT AND FILES #####
        base_path = f"runs/{run_family}/{run_name}/{n_samples}/train/val_{k}/"
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
            net = SONCLIPFewShotNet(model, exp_params['hyperparams']['coop'], use_embeddings=False)

        model = CLIPFewShotModule(organizer.root_path+'outputs/', net, run_name, label_info, exp_params['descriptions']['splits'])
        model.set_hyperparams(exp_params['hyperparams']['optim']['lr'], exp_params['hyperparams']['optim']['decay'])

        checkpoint_callback = ModelCheckpoint(
            monitor= 'val_mse',
            dirpath= str(organizer.states_path),
            filename='{epoch:02d}-{mse:.2f}',
            save_top_k=1,
            mode='min')

        ##### SETUP TRAINER #####
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
        
        # # Enable loading of embeddings
        model.net.coop_learner.use_embeddings = True

        data_module.setup_data_classes( 
                                        data_container,
                                        exp_params['paths']['images_root'],
                                        split_ids,
                                        embeddings=exp_params['paths']['embeddings'],
                                        transforms=transforms,
                                        id_col=exp_params['descriptions']['id_col'],
                                        splits=exp_params['descriptions']['splits']
                                    )

        for i, param in enumerate(model.parameters()):
            param.requires_grad_(False)

        best_model = next(Path(f"runs/{run_family}/{run_name}/{n_samples}/train/val_{k}/outputs/states/").glob('**/*'))
        state = torch.load(best_model)['prompter']
        model.load_state_dict(state, strict=False)    
        
        tester = Trainer(gpus=exp_params['hyperparams']['gpu_nums'], logger=tb_logger)
        tester.test(model, datamodule=data_module) 