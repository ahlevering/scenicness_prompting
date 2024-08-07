import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl
from codebase.experiment_tracking.run_tracker import VarTrackerCLIPExperiments

import torch.nn.functional as F

class BaselineModel(pl.LightningModule):
    def __init__(self, scatter_dir, net, run_name, label_info, embeddings_policy, splits, tracker_class):
        super().__init__()
        self.net = net
        self.label_info = label_info

        self.run_name = run_name
        self.out_dir = scatter_dir
        self.embeddings_policy = embeddings_policy

        if 'train' in splits:
            self.train_tracker = VarTrackerCLIPExperiments(self.out_dir, 'train', label_info, tracker_class)
        if 'val' in splits:
            self.val_tracker = VarTrackerCLIPExperiments(self.out_dir, 'val', label_info, tracker_class)
        if 'test' in splits or 'all' in splits:
            self.test_tracker = VarTrackerCLIPExperiments(self.out_dir, 'test', label_info, tracker_class)            

### General iteration functions ###
    def forward(self, x):
        x = self.net(x)
        return x

    def iteration_forward(self, batch, tracker, split):                
        preds = self.net(batch['img'])

        loss = F.mse_loss(preds.double().squeeze(), batch['gt'].double().squeeze())

        ## Get metadata
        ids = batch['ids'].cpu().numpy()
        lat = batch['lat'].cpu().numpy()
        lon = batch['lon'].cpu().numpy()
        preds_out = preds.detach().cpu().numpy()
        gt_out = batch['gt'].detach().cpu().numpy()

        ## Create storage dicts
        datapts = {'ids':ids, 'lat':lat, 'lon':lon, 'preds':preds_out, 'gt':gt_out}

        ## Store to trackers
        self.datapoints_to_tracker(tracker, datapts, 'scenic')
        self.log(f'{split}_mse', loss.detach().cpu(), on_step=True, on_epoch=True)        
        return loss

    def datapoints_to_tracker(self, tracker, datapts, var_name):
        length_is_one = len(datapts['preds']) == 1
        for attr in datapts:
            # if length_is_one:
            #     datapts[attr] = datapts[attr]
            #     # ids = [int(ids)]         
            # elif type(datapts[attr]) != list:
            # datapts[attr] = datapts[attr].squeeze()
            
            # Store into tracker
            datapts[attr] = datapts[attr].tolist()
            tracker.variables[var_name].attrs[attr].extend(datapts[attr])

        # tracker.variables[var_name].vars['ids'].extend(ids)

    def end_epoch(self, tracker, store_outputs=False):
        tracker.store_epoch_metrics()
        
        ## Write outputs
        tracker.save_metrics_to_file()
        tracker.save_observations_to_file(self.current_epoch)
        # tracker.save_scatterplot(self.current_epoch)

        ## Reset for next epoch
        if store_outputs:
            tracker.save_observations_to_file(self.current_epoch)
        tracker.reset_epoch_vars()
        # tracker.print_results()

### Training ###
    def on_train_epoch_start(self):
        self.net.use_embeddings = self.embeddings_policy['train']

    def training_step(self, batch, batch_idx):
        loss = self.iteration_forward(batch, self.train_tracker, 'train')
        return loss

    def on_train_epoch_end(self):
        self.end_epoch(self.train_tracker)
        if 'scenic' in self.train_tracker.variables:
            self.log('train_r2', self.train_tracker.variables['scenic'].metrics['rsquared'][-1])        
        if self.current_epoch == 0:
            for param in self.net.parameters():
                param.requires_grad = True
            
            
# ### Validation ###
    def on_validation_epoch_start(self):
        self.net.use_embeddings = self.embeddings_policy['val']
        ## Clear out val test run data from quick dev runs ##
        if self.current_epoch == 0:
            self.val_tracker.reset_out_files()   

    def validation_step(self, batch, batch_idx):
        loss = self.iteration_forward(batch, self.val_tracker, 'val')
        return loss

    def on_validation_epoch_end(self):
        self.end_epoch(self.val_tracker)
        if 'scenic' in self.val_tracker.variables:
            self.log('val_r2', self.val_tracker.variables['scenic'].metrics['rsquared'][-1])
        else:
            pass # all_r2s = [tracker.metrics['rsquared'][-1] for tracker in self.val_tracker.variables.values() if not len(tracker.metrics['rsquared']) == 0]

### Testing ###
    def on_test_epoch_start(self):
        self.net.use_embeddings = self.embeddings_policy['test']

    def test_step(self, batch, batch_idx):
        loss = self.iteration_forward(batch, self.test_tracker, 'test')
        return loss

    def on_test_epoch_end(self):
        self.end_epoch(self.test_tracker, store_outputs=False)
    
    def set_hyperparams(self, lr=0.0001, decay=0.0001):
        self.lr = lr
        self.decay = decay

    def configure_optimizers(self):
        if "baseline" in self.run_name:        
            # optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.decay)
            optimizer = torch.optim.AdamW(
                [
                    # {"params": self.net.vision_model.parameters(), "lr": self.lr/10},
                    {"params": self.net.fc.parameters(), "lr": self.lr*10},
                ],
                lr=self.lr
            )            
        scheduler1 = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)

        return [optimizer], [scheduler1]