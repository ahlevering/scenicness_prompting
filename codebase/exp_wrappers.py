from pathlib import Path
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from codebase.experiment_tracking.save_metadata import ExperimentOrganizer
from codebase.experiment_tracking import process_yaml
from codebase.pt_funcs.dataloaders import ClipDataLoader
# from codebase.pt_funcs.models_few_shot import CLIPMulticlassFewShotNet, SONLinearProbe, CLIPMultiPromptNet
from codebase.pt_funcs.models_zero_shot import MultiClassPrompter, MultiClassContrasts
# from codebase.pt_funcs.models_baseline import ConvNext_regression

from codebase.utils.file_utils import load_csv, make_crossval_splits

def get_label_embeds_paths(exp_params):
    if exp_params['descriptions']['debug']:
        embeddings = exp_params['paths']['debug']
        labels = exp_params['paths']['debug']['debug_labels_file']
        embeddings = exp_params['paths']['debug']['debug_embeds_file']
    else:
        labels = exp_params['paths']['labels_file']
        embeddings = exp_params['paths']['embeddings_file']
    return labels, embeddings

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
        n_workers = 3
        batch_size = 1
    else:
        n_workers = exp_params['hyperparams']['workers']
        batch_size = exp_params['hyperparams']['batch_size']
    data_module = ClipDataLoader(n_workers, 
                                 batch_size,
                                 data_class=data_class)

    data_module.setup_data_classes(data_container,
                                   exp_params['paths']['images_root'],
                                   split_ids,
                                   embeddings_policy=exp_params['hyperparams']['use_precalc_embeddings'],
                                   transforms=transforms,
                                   splits=exp_params['descriptions']['splits'])
    # loader_emulator = next(iter(data_module.train_data)) # Test if loader can return a batch
    return data_module

def setup_trainer(organizer, run_name, exp_params, to_monitor="val_r2", monitor_mode='max'):
    checkpoint_callback = ModelCheckpoint(monitor=to_monitor,
                                          dirpath=str(organizer.states_path),
                                          filename='{epoch:02d}-{val_r2:.4f}',
                                          save_top_k=8,
                                          mode=monitor_mode)

    ##### SETUP TRAINER #####
    logger = WandbLogger(save_dir=str(organizer.logs_path), name=run_name)
    trainer = Trainer(max_epochs=exp_params['hyperparams']['epochs'],
                      devices=1,
                      accelerator="gpu",                      
                      callbacks=[checkpoint_callback],
                      logger = logger,
                      limit_train_batches = 0.5,
                      fast_dev_run=False)
    return trainer

def setup_model(backbone, exp_params):
    backbone = backbone
    model_type = exp_params["descriptions"]["model_type"]

    # Freeze feature extractors
    if model_type not in ["unfrozen_probe"]:
        for i, param in enumerate(backbone.parameters()):
            param.requires_grad_(False)

    elif model_type in ["prompt_learner_multiclass"]:
        net = CLIPMulticlassFewShotNet(backbone, exp_params['hyperparams']['coop'])        
    elif model_type in ["multiprompt"]:
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in list(exp_params['hyperparams']['prompts'].keys())]).to(exp_params['hyperparams']['gpu_nums'][0])
        prompt_values = torch.tensor(list(exp_params['hyperparams']['prompts'].values())).to(exp_params['hyperparams']['gpu_nums'][0])
        son_rescale = 'son' in exp_params['descriptions']['exp_family']
        net = CLIPMultiPromptNet(backbone, tokenized_prompts, prompt_values, son_rescale=son_rescale)
    elif model_type in ["linear_probe", "unfrozen_probe"]:
        backbone = backbone.visual
        net = SONLinearProbe(backbone)
    elif model_type == "frozen_multiprompt": # Technical debt, can be in few-shot, solve differently?
        # lc_rows = load_csv(exp_params['hyperparams']['lut_file'])
        # lc_class_names = [x[-1] for x in lc_rows]
        class_list = list(exp_params['hyperparams']['class_list'].values())
        prompts = [f"{exp_params['hyperparams']['prompt']} {x}" for x in class_list]
        net = MultiClassPrompter(backbone, prompts)
    elif model_type == "contrastive_extractor":
        prompt_contrasts = exp_params['hyperparams']['prompt_contrast']
        prompts = exp_params['hyperparams']['prompts']
        pos_prompts = [f"{prompt_contrasts[0]} {p}." for p in prompts]
        neg_prompts = [f"{prompt_contrasts[1]} {p}." for p in prompts]
        net = MultiClassContrasts(backbone, neg_prompts, pos_prompts)
    return net

def setup_label_info(keys):
    # More of a placeholder than anything, should be fixed
    label_info = {}    
    for key in keys:
        label_info[key] = {}
        label_info[key]['ylims'] = [1, 10]
    return label_info

def setup_pp2_label_info():
    label_info = {}    
    for label in ["lively", "depressing" , "boring", "beautiful", "safety", "wealthy"]:    
        label_info[label] = {}
        label_info[label]['index'] = 0
        label_info[label]['ylims'] = [0, 1]
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