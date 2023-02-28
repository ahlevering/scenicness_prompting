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
# setup_file = "setup_files/train/son_baseline.yaml"
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
for n_samples in [25, 50, 100, 250, 500]:
    if n_samples <= 100:
        exp_params['hyperparams']['coop']['ctx_init'] = exp_params['hyperparams']['coop']['ctx_init_low']
    else:
        exp_params['hyperparams']['coop']['ctx_init'] = exp_params['hyperparams']['coop']['ctx_init_high']
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

        for architecture in ["RN50", "ViT-L/14"]:
            archi_save_name = architecture.replace("/", "-") # Why did they use slashes in their naming!?
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
            base_path = f"runs/{run_family}/{run_name}/{archi_save_name}/{n_samples}/val_{k}/"
            organizer = ExperimentOrganizer(base_path)
            organizer.store_yaml(setup_file)
            organizer.store_environment()
            organizer.store_codebase(['.py'])

            ##### SETUP MODEL #####
            model, preprocess = clip.load(architecture)      

            # Freeze feature extractors
            if not 'unfrozen' in run_name:
                for i, param in enumerate(model.parameters()):
                    param.requires_grad_(False)
            # Load model class
            if 'linear_probe' in run_name:
                net = CLIPLinearProbe(model)
            elif 'baseline' in run_name:
                net = Baseline(model, use_embeddings=False)
            else:    
                net = SONCLIPFewShotNet(model, exp_params['hyperparams']['coop'], use_embeddings=False)

            # Setup training wrapper
            model = CLIPFewShotModule(organizer.root_path+'outputs/', net, run_name, label_info, exp_params['descriptions']['splits'])
            model.set_hyperparams(exp_params['hyperparams']['optim']['lr'], exp_params['hyperparams']['optim']['decay'])

            checkpoint_callback = ModelCheckpoint(
                monitor= 'val_r2',
                dirpath= str(organizer.states_path),
                filename='{epoch:02d}-{val_r2:.4f}',
                save_top_k=3,
                mode='max')

            ##### SETUP TRAINER #####
            tb_logger = TensorBoardLogger(save_dir=organizer.logs_path, name=run_name)
            trainer = Trainer(  max_epochs=exp_params['hyperparams']['epochs'],
                                gpus=exp_params['hyperparams']['gpu_nums'],
                                callbacks=[checkpoint_callback],
                                logger = tb_logger,
                                fast_dev_run=False,
                                # limit_train_batches=250,
                                # limit_val_batches=10,
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
                                            embeddings=exp_params['paths']['embeddings_root']+f"{archi_save_name}.pkl",
                                            transforms=transforms,
                                            id_col=exp_params['descriptions']['id_col'],
                                            splits=exp_params['descriptions']['splits']
                                        )

            for i, param in enumerate(model.parameters()):
                param.requires_grad_(False)

            all_states = list(Path(f"runs/{run_family}/{run_name}/{archi_save_name}/{n_samples}/val_{k}/outputs/states/").glob('**/*'))
            top_model_metric = sorted([float(str(s).split("=")[-1].split(".ckpt")[0]) for s in all_states])[-1]
            top_model_path = [s for s in all_states if str(top_model_metric) in str(s)][0]

            state = torch.load(top_model_path)['prompter']
            model.load_state_dict(state, strict=False)    
            
            tester = Trainer(gpus=exp_params['hyperparams']['gpu_nums'], logger=tb_logger)
            tester.test(model, datamodule=data_module) 