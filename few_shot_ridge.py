import yaml
from pathlib import Path

import clip
import torch
import numpy as np
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.seed import seed_everything
from sklearn.linear_model import Ridge

from codebase.pt_funcs.dataloaders import SONData, ClipDataLoader, SoNDataContainer, load_pickle
from codebase.pt_funcs.models_few_shot import ContrastiveManyPromptsNet, CLIPFewShotModule
from codebase.utils.file_utils import load_csv

from codebase.experiment_tracking import process_yaml
from codebase.experiment_tracking.save_metadata import ExperimentOrganizer
    
##### SET GLOBAL OPTIONS ######
seed_everything(113)
np.set_printoptions(suppress=True)
torch.set_printoptions(sci_mode=False)

##### LOAD SET-UP FILE #####
setup_file = "setup_files/train/son_few_shot_regression.yaml"
with open(setup_file) as file:
    exp_params = yaml.full_load(file)

run_name = exp_params['descriptions']['name']
run_family = exp_params['descriptions']['exp_family']

# Tokenize prompts
prompts = list(exp_params['hyperparams']['coop']['prompts'].keys())
prompts = torch.cat([clip.tokenize(p+".") for p in prompts])
prompts = prompts.to(device=exp_params['hyperparams']['gpu_nums'][0])

# Make weights learnable
architecture = "ViT-L/14"
archi_save_name = architecture.replace("/", "-") # Why did they use slashes in their naming!?

embeddings_path = exp_params['paths']['embeddings_root']+f"{archi_save_name}.pkl"
embeddings = load_pickle(embeddings_path)
prompt_weights = torch.tensor(list(exp_params['hyperparams']['coop']['prompts'].values())).to(device=exp_params['hyperparams']['gpu_nums'][0])

data_container = SoNDataContainer(exp_params['paths']['labels_file'])
labels = data_container.labels

net, preprocess = clip.load(architecture)
net = net.to(device=exp_params['hyperparams']['gpu_nums'][0])
model = ContrastiveManyPromptsNet(net, prompts, prompt_weights, use_embeddings=True)

with torch.no_grad():
    for n_samples in [25, 50, 100, 250, 500]:
        all_split_ids = load_csv(exp_params['paths']['splits_root']+f'{n_samples}.csv')
        split_gt = labels[labels['ID'].isin(all_split_ids)]['mean']
        split_embeddings = [embeddings[str(k)].to(device=exp_params['hyperparams']['gpu_nums'][0]) for k in all_split_ids]
        activations = [model.forward(s).detach().cpu().numpy().squeeze() for s in split_embeddings]
        
        clf = Ridge(alpha=0.1)
        model = clf.fit(activations, split_gt)


    #         ##### SETUP LOADERS #####
    #         data_module = ClipDataLoader( exp_params['hyperparams']['workers'],    
    #                                     exp_params['hyperparams']['batch_size'],
    #                                     data_class=SONData
    #                                     )

    #         data_module.setup_data_classes( data_container,
    #                                         exp_params['paths']['images_root'],
    #                                         split_ids,
    #                                         embeddings=None,
    #                                         transforms=transforms,
    #                                         id_col=exp_params['descriptions']['id_col'],
    #                                         splits=['train', 'val']
    #                                     )

    #         loader_emulator = next(iter(data_module.train_data))

    #         ##### STORE ENVIRONMENT AND FILES #####
    #         base_path = f"runs/{run_family}/{run_name}/{n_samples}/{lr}/val_{k}/"
    #         organizer = ExperimentOrganizer(base_path)
    #         organizer.store_yaml(setup_file)
    #         organizer.store_environment()
    #         organizer.store_codebase(['.py'])

    #         ##### SETUP MODEL #####
    #         model, preprocess = clip.load(architecture)

    #         # Freeze feature extractors
    #         if not 'unfrozen' in run_name:
    #             for i, param in enumerate(model.parameters()):
    #                 param.requires_grad_(False)
    #         # Load model class
    #         net = ContrastiveManyPromptsNet(model, prompts, values, use_embeddings=False)

    #         # Setup training wrapper
    #         model = CLIPFewShotModule(organizer.root_path+'outputs/', net, run_name, label_info, ['train', 'val'])
    #         model.set_hyperparams(lr, exp_params['hyperparams']['optim']['decay'])

    #         checkpoint_callback = ModelCheckpoint(
    #             monitor= 'val_r2',
    #             dirpath= str(organizer.states_path),
    #             filename='{epoch:02d}-{val_r2:.4f}',
    #             save_top_k=8,
    #             mode='max')

    #         ##### SETUP TRAINER #####
    #         tb_logger = TensorBoardLogger(save_dir=organizer.logs_path, name=run_name)
    #         trainer = Trainer(  max_epochs=exp_params['hyperparams']['epochs'],
    #                             gpus=exp_params['hyperparams']['gpu_nums'],
    #                             callbacks=[checkpoint_callback],
    #                             logger = tb_logger,
    #                             fast_dev_run=False,
    #                             # limit_train_batches=250,
    #                             # limit_val_batches=10,
    #                         )

    #         ##### FIT MODEL #####
    #         print(f"fitting {run_name}")
    #         trainer.fit(model, datamodule=data_module)            
    #         k_rsquared.append(model.val_tracker.variables['scenic'].metrics['rsquared'][-1])

    # ##### Test Model #####

    # ##### STORE ENVIRONMENT AND FILES #####
    # base_path = f"runs/{run_family}/{run_name}/{n_samples}/best/"
    # organizer = ExperimentOrganizer(base_path)
    # organizer.store_yaml(setup_file)
    # organizer.store_environment()
    # organizer.store_codebase(['.py'])    
    # split_ids = {'train': all_split_ids, 'val': all_split_ids, 'test': test_ids}

    # # Get hyperparam performance
    # best_index = np.argmax(lr_rsquared)
    # lr = exp_params['hyperparams']['optim']['lr'][best_index]

    # best_states_dir = f"runs/{run_family}/{run_name}/{n_samples}/{lr}/val_{best_index}/"
    # states_of_best_hparams = list(Path(f"{best_states_dir}outputs/states/").glob('**/*'))
    # # Only consider models after the first 5 epochs for stability of low number of samples
    # converged_states = [s for s in states_of_best_hparams if int(str(s).split("/")[-1][6:8]) >= 0]
    # top_model_metric = sorted([float(str(s).split("=")[-1].split(".ckpt")[0]) for s in converged_states])[-1] # Take best model
    # top_model_path = [s for s in converged_states if str(top_model_metric) in str(s)][0]   


    # ##### SETUP MODEL #####
    # model, preprocess = clip.load(architecture)

    # data_module = ClipDataLoader(   6,    
    #                                 1,
    #                                 data_class=SONData
    #                             )

    # data_module.setup_data_classes( 
    #                                 data_container,
    #                                 exp_params['paths']['images_root'],
    #                                 split_ids,
    #                                 embeddings=None, # exp_params['paths']['embeddings_root']+f"{archi_save_name}.pkl",
    #                                 transforms=transforms,
    #                                 id_col=exp_params['descriptions']['id_col'],
    #                                 splits=['train', 'val', 'test'] 
    #                             )    

    # # Freeze feature extractors
    # if not 'unfrozen' in run_name:
    #     for i, param in enumerate(model.parameters()):
    #         param.requires_grad_(False)
    # # Load model class
    # net = ContrastiveManyPromptsNet(model, prompts, values, use_embeddings=False)

    # state = torch.load(top_model_path)['prompter']
    # model.load_state_dict(state, strict=False)        

    # # Setup training wrapper
    # model = CLIPFewShotModule(organizer.root_path+'outputs/', net, run_name, label_info, ['train', 'val', 'test'])
    # model.set_hyperparams(lr, exp_params['hyperparams']['optim']['decay'])

    # checkpoint_callback = ModelCheckpoint(
    #     monitor= 'val_r2',
    #     dirpath= str(organizer.states_path),
    #     filename='{epoch:02d}-{train_r2:.4f}',
    #     save_top_k=8,
    #     mode='max')   

    # ##### SETUP TRAINER #####
    # tb_logger = TensorBoardLogger(save_dir=organizer.logs_path, name=run_name)
    # trainer = Trainer(  max_epochs=exp_params['hyperparams']['epochs'],
    #                     gpus=exp_params['hyperparams']['gpu_nums'],
    #                     callbacks=[checkpoint_callback],
    #                     logger = tb_logger,
    #                     fast_dev_run=False,                
    #                     # limit_train_batches=250,
    #                     # limit_val_batches=0.0,
    #                 )

    # ##### FIT MODEL #####
    # print(f"fitting {run_name}")
    # trainer.fit(model, datamodule=data_module)
    
    # ##### TEST MODEL #####
    # # # Enable loading of embeddings

    # data_module = ClipDataLoader(   8,    
    #                                 64,
    #                                 data_class=SONData
    #                             ) 

    # data_module.setup_data_classes( 
    #                                 data_container,
    #                                 exp_params['paths']['images_root'],
    #                                 split_ids,
    #                                 embeddings=exp_params['paths']['embeddings_root']+f"{archi_save_name}.pkl",
    #                                 transforms=transforms,
    #                                 id_col=exp_params['descriptions']['id_col'],
    #                                 splits=['test'] 
    #                             )
    
    # model.net.use_embeddings = True

    # for i, param in enumerate(model.parameters()):
    #     param.requires_grad_(False)

    # all_states = list(Path(f"{base_path}outputs/states/").glob('**/*'))
    # # Only consider models after the first 5 epochs for stability of low number of samples
    # converged_states = [s for s in all_states if int(str(s).split("/")[-1][6:8]) >= 5]
    # top_model_metric = sorted([float(str(s).split("=")[-1].split(".ckpt")[0]) for s in converged_states])[-1] # Take best model
    # top_model_path = [s for s in all_states if str(top_model_metric) in str(s)][0]

    # state = torch.load(top_model_path)['prompter']
    # model.load_state_dict(state, strict=False)    
    
    # tester = Trainer(gpus=exp_params['hyperparams']['gpu_nums'], logger=tb_logger)
    # tester.test(model, datamodule=data_module)
