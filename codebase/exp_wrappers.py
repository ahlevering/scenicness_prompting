from pathlib import Path

import clip
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

from codebase.experiment_tracking.save_metadata import ExperimentOrganizer
from codebase.experiment_tracking import process_yaml
from codebase.pt_funcs.dataloaders import ClipDataLoader
from codebase.pt_funcs.models_few_shot import SONCLIPFewShotNet, SONBaseline

from codebase.utils.file_utils import load_csv, make_crossval_splits

def setup_split_transforms(exp_params):
    transforms = {}
    for split in exp_params['transforms']:
        transforms[split] = process_yaml.setup_transforms(exp_params['transforms'][split])
    return transforms

def organize_experiment_info(run_path, setup_file):
    organizer = ExperimentOrganizer(run_path)
    organizer.store_yaml(setup_file)
    organizer.store_environment()
    organizer.store_codebase(['.py'])
    return organizer

def setup_data_module(data_class, exp_params, data_container, transforms, split_ids=None, single_batch=False):
    if single_batch:
        n_workers = 8
        batch_size = 1
    else:
        n_workers = exp_params['n_workers']
        n_workers = exp_params['batch_size']
    data_module = ClipDataLoader(n_workers, 
                                 batch_size,
                                 data_class=data_class)

    data_module.setup_data_classes(data_container,
                                   exp_params['paths']['images_root'],
                                   split_ids,
                                   transforms=transforms,
                                   id_col=exp_params['descriptions']['id_col'],
                                   splits=exp_params['descriptions']['splits'])
    loader_emulator = next(iter(data_module.train_data)) # Check if loader can return a batch
    return data_module

def setup_trainer(organizer, run_name, exp_params, to_monitor="val_r2", monitor_mode='max'):
    checkpoint_callback = ModelCheckpoint(monitor=to_monitor,
                                          dirpath=str(organizer.states_path),
                                          filename='{epoch:02d}-{val_r2:.4f}',
                                          save_top_k=8,
                                          mode=monitor_mode)

    ##### SETUP TRAINER #####
    tb_logger = TensorBoardLogger(save_dir=organizer.logs_path, name=run_name)
    trainer = Trainer(max_epochs=exp_params['hyperparams']['epochs'],
                      gpus=exp_params['hyperparams']['gpu_nums'],
                      callbacks=[checkpoint_callback],
                      logger = tb_logger,
                      fast_dev_run=False)
    return trainer

def setup_model(backbone, exp_params, model_type="learned_prompt_context", use_embeddings=True):
    ##### SETUP MODEL #####
    backbone, _ = clip.load(backbone)

    # Freeze feature extractors
    if model_type in ["learned_prompt_context"]:
        for i, param in enumerate(backbone.parameters()):
            param.requires_grad_(False)
    if model_type == "learned_prompt_context":
        net = SONCLIPFewShotNet(backbone, exp_params['hyperparams']['coop'], use_embeddings=use_embeddings)
    elif model_type == "probe":
        net = SONCLIPFewShotNet(backbone, exp_params['hyperparams']['coop'], use_embeddings=use_embeddings)
    return net

def setup_son_label_info():
    label_info = {}
    label_info['scenic'] = {}
    label_info['scenic']['ylims'] = [1, 10]
    return label_info

def get_sample_split_ids(exp_params, n_samples, to_int=False):
    all_split_ids = load_csv(exp_params['paths']['splits_root']+f'{n_samples}.csv')[0]
    if to_int:
        all_split_ids = [int(i) for i in all_split_ids] # Badness, clean later in data itself
    return all_split_ids

def get_crossval_splits(all_split_ids, k_folds, data_container):
    train_ids, val_ids = make_crossval_splits(all_split_ids, k_folds) # Subdivide pre-sampled indices into k-fold bins
    test_ids = data_container.labels[~data_container.labels['ID'].isin(all_split_ids)]
    test_ids = list(test_ids['ID'].values)
    return train_ids, val_ids, test_ids

def get_top_model(best_states_dir, min_epoch=0):
    states_of_best_hparams = list(Path(f"{best_states_dir}outputs/states/").glob('**/*'))
    # Only consider models after the first 5 epochs for stability of low number of samples
    converged_states = [s for s in states_of_best_hparams if int(str(s).split("/")[-1][6:8]) >= min_epoch]
    # Dirty way to identify the model with the best performance
    top_model_metric = sorted([float(str(s).split("=")[-1].split(".ckpt")[0]) for s in converged_states])[-1] # Take best model
    top_model_path = [s for s in converged_states if str(top_model_metric) in str(s)][0]
    return top_model_path

# def setup_architecture(backbone, freeze=False):
#     model, _ = clip.load(architecture)      

#     if freeze:
#         for i, param in enumerate(model.parameters()):
#             param.requires_grad_(False)
#     net = SONCLIPFewShotNet(model, exp_params['hyperparams']['coop'], use_embeddings=True)