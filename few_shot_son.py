import yaml

import torch
import numpy as np
from sklearn.model_selection import train_test_split
from pytorch_lightning.utilities.seed import seed_everything

from codebase.pt_funcs.dataloaders import SoNDataContainer, SONData
from codebase.pt_funcs.models_few_shot import CLIPFewShotModule
from codebase.experiment_tracking.run_tracker import VarTrackerRegression

from codebase.exp_wrappers import *

##### SET GLOBAL OPTIONS ######
torch.manual_seed(113)
np.set_printoptions(suppress=True)
torch.set_printoptions(sci_mode=False)

##### LOAD SET-UP FILE #####
setup_file = "setup_files/train/son_linear_probe.yaml"
# setup_file = "setup_files/train/son_few_shot_imgnet.yaml"
# setup_file = "setup_files/train/son_coop_contrastive.yaml"
with open(setup_file) as file:
    exp_params = yaml.full_load(file)

labels_path, embeddings_path = get_label_embeds_paths(exp_params)
data_container = SoNDataContainer(labels_path, embeddings_path)

train_val_labels, test_labels = train_test_split(data_container.labels["ID"], test_size=0.1, random_state=113)
train_labels, val_labels = train_test_split(train_val_labels, test_size=0.111, random_state=113)


backbone = "ViT-L/14"
run_name = exp_params["descriptions"]["name"]
run_family = exp_params["descriptions"]["exp_family"]
k_folds = exp_params["hyperparams"]["k_folds"]

label_info = setup_label_info(["scenic"])

##### SET UP TRANSFORMS #####
transforms = setup_split_transforms(exp_params)

# prompts = exp_params['hyperparams']['prompts']
# prompts = torch.cat([clip.tokenize(p) for p in prompts])
# prompts = prompts.to(device=exp_params['hyperparams']['gpu_nums'][0])

############
# TRAINING #
############

state = None
for n_samples in [25, 50, 100, 250, 500]:
    if n_samples >= 250:
        min_epoch = 0
    else:
        min_epoch = 6  # Only load states from after the 6th epoch to avoid unstable models
    sample_split_ids = get_sample_split_ids(exp_params, n_samples, to_int=True)

    lr_rsquared = []
    for lr in exp_params["hyperparams"]["optim"]["lr"]:
        run_file_storage_path = f"runs/{run_family}/{run_name}/{n_samples}/{lr}/"
        if k_folds > 1:
            k_rsquared = []
            for k in range(k_folds):
                train_ids, val_ids, test_ids = get_crossval_splits(sample_split_ids, k_folds, data_container)
                split_k_ids = {"train": train_ids[k], "val": val_ids[k], "test": None}

                ##### STORE ENVIRONMENT AND FILES #####
                storage_path_k = run_file_storage_path + f"/val_{k}/"
                organizer = organize_experiment_info(storage_path_k, setup_file)

                ##### SETUP DATASET #####
                data_module = setup_data_module(
                    SONData, exp_params, data_container, transforms, split_ids=split_k_ids, single_batch=True
                )

                ##### SETUP MODEL #####
                net = setup_model(backbone, exp_params)
                model = CLIPFewShotModule(
                    organizer.root_path + "outputs/",
                    net,
                    run_name,
                    label_info,
                    exp_params["hyperparams"]["use_precalc_embeddings"],
                    ["train", "val"],
                    VarTrackerRegression,
                )
                model.set_hyperparams(lr, exp_params["hyperparams"]["optim"]["decay"])

                ##### SETUP TRAINER #####
                trainer = setup_trainer(organizer, run_name, exp_params, to_monitor="val_r2", monitor_mode="max")

                ##### FIT MODEL #####
                trainer.fit(model, datamodule=data_module)
                k_rsquared.append(model.val_tracker.variables["scenic"].metrics["rsquared"][-1])

            # Aggregate metric over all k splits
            lr_rsquared.append(np.mean(k_rsquared))

    ############################
    # RE-TRAIN FROM BEST STATE #
    ############################
    run_file_storage_path = f"runs/{run_family}/{run_name}/{n_samples}/{lr}/"
    if k_folds > 1:
        ##### GET BEST HYPERPARAMETER & STATE #####
        best_index = np.argmax(lr_rsquared)
        lr = exp_params["hyperparams"]["optim"]["lr"][best_index]

        best_states_dir = f"runs/{run_family}/{run_name}/{n_samples}/{lr}/val_{best_index}/"
        top_model_path = get_top_model(best_states_dir, min_epoch)
        state = torch.load(top_model_path)["state"]
        run_file_storage_path += "best/"

    test_ids = data_container.labels[~data_container.labels["ID"].isin(sample_split_ids)]

    ##### STORE ENVIRONMENT AND FILES #####
    organizer = organize_experiment_info(run_file_storage_path, setup_file)
    split_ids = {"train": sample_split_ids, "val": sample_split_ids, "test": test_ids["ID"].values.tolist()}

    ##### SETUP DATASET #####
    data_module = setup_data_module(
        SONData, exp_params, data_container, transforms, split_ids=split_ids, single_batch=True
    )

    ##### SETUP MODEL #####
    net = setup_model(backbone, exp_params)
    if state:
        net.load_state_dict(state, strict=False)
    model = CLIPFewShotModule(
        organizer.root_path + "outputs/",
        net,
        run_name,
        label_info,
        exp_params["hyperparams"]["use_precalc_embeddings"],
        ["train", "val", "test"],
        VarTrackerRegression,
    )
    model.set_hyperparams(lr, exp_params["hyperparams"]["optim"]["decay"])

    ##### SETUP TRAINER #####
    trainer = setup_trainer(organizer, run_name, exp_params, to_monitor="val_r2", monitor_mode="max")

    ##### FIT MODEL #####
    trainer.fit(model, datamodule=data_module)

    ###################
    # TEST BEST MODEL #
    ###################
    # Turn off single-batch loading
    data_module = setup_data_module(
        SONData, exp_params, data_container, transforms, split_ids=split_ids, single_batch=False
    )

    for i, param in enumerate(model.parameters()):
        param.requires_grad_(False)

    top_model_path = get_top_model(run_file_storage_path, min_epoch)

    state = torch.load(top_model_path)["state"]
    model.load_state_dict(state, strict=False)

    tb_logger = TensorBoardLogger(save_dir=organizer.logs_path, name=run_name)
    tester = Trainer(gpus=exp_params["hyperparams"]["gpu_nums"], logger=tb_logger)
    tester.test(model, datamodule=data_module)
