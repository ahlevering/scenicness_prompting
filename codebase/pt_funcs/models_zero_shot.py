from copy import deepcopy

import torch
from torch import nn
import pytorch_lightning as pl
from codebase.experiment_tracking.run_tracker import VarTrackerCLIPExperiments

class MultiClassPrompter(nn.Module):
    def __init__(self, model, prompts, use_embeddings=False):
        super().__init__()
        self.use_embeddings = use_embeddings
        self.model = model
        device_num = torch.get_device(next(iter(model.visual.parameters())))        
        tokenized_prompts = clip.tokenize(prompts).to(device_num)
        self.txt_feats = self.model.encode_text(tokenized_prompts)
        self.txt_feats /= self.txt_feats.norm(dim=-1, keepdim=True)

    def get_prompt_activations(self, img_feats):
        img_feats /= img_feats.norm(dim=-1, keepdim=True)
        activations = (100.0 * img_feats @ self.txt_feats.T)
        softmax_scores = activations.softmax(dim=1)
        return softmax_scores

    def forward(self, img):
        if self.use_embeddings:
            img_feats = img
        else:
            img_feats = self.model.visual(img)
        scoring = self.get_prompt_activations(img_feats)
        return scoring

class MultiClassContrasts(nn.Module):
    def __init__(self, model, negative_prompts, positive_prompts, use_embeddings=False):
        super().__init__()
        self.use_embeddings = use_embeddings
        self.model = model
        device_num = torch.get_device(next(iter(model.visual.parameters())))        
        # Negative prompts
        neg_tokenized_prompts = clip.tokenize(negative_prompts).to(device_num)
        self.neg_txt_feats = self.model.encode_text(neg_tokenized_prompts)
        self.neg_txt_feats /= self.neg_txt_feats.norm(dim=-1, keepdim=True)
        # Positive prompts
        pos_tokenized_prompts = clip.tokenize(positive_prompts).to(device_num)
        self.pos_txt_feats = self.model.encode_text(pos_tokenized_prompts)
        self.pos_txt_feats /= self.pos_txt_feats.norm(dim=-1, keepdim=True)   

    def get_prompt_activations(self, img_feats):
        img_feats /= img_feats.norm(dim=-1, keepdim=True)
        pos_activations = (100.0 * img_feats @ self.pos_txt_feats.T)
        neg_activations = (100.0 * img_feats @ self.neg_txt_feats.T)

        # Flatten the tensor back to its original shape
        softmax_scores = nn.functional.softmax(torch.stack((pos_activations, neg_activations), dim=2), dim=2)
        # pos_activations = softmax_scores[:, :, 0]
        return pos_activations

    def forward(self, img):
        if self.use_embeddings:
            img_feats = img
        else:
            img_feats = self.model.visual(img)
        scoring = self.get_prompt_activations(img_feats)
        return scoring        

class PP2ManyPrompts(nn.Module):
    def __init__(self, model, prompts, prompt_values, use_embeddings=False, son_rescale=False):
        super().__init__()
        self.model = model
        self.prompts = prompts
        self.prompt_values = prompt_values
        self.use_embeddings=use_embeddings
        self.son_rescale = True

    def simple_contrastive(self, img_feats, txt_feats):
        img_feats /= img_feats.norm(dim=-1, keepdim=True)
        txt_feats /= txt_feats.norm(dim=-1, keepdim=True)
        activations = (100.0 * img_feats @ txt_feats.T)# .softmax(dim=-1)
        output = activations * self.prompt_values.unsqueeze(0)
        return output.mean(-1)

    def get_image_score(self, img, txt_feats):
        if self.use_embeddings:
            img_feats = img
        else:
            img_feats = self.model.visual(img)
        txt_feats = self.model.encode_text(self.prompts)
        scoring = self.simple_contrastive(img_feats, txt_feats)
        return scoring.unsqueeze(-1)

    def forward(self, imgs, indices):
        txt_feats = self.model.encode_text(self.prompts)                
        if type(imgs) == dict:# > 1:
            img1_scores = self.get_image_score(imgs['img_left'], txt_feats).repeat([1,6])
            img2_scores = self.get_image_score(imgs['img_right'], txt_feats).repeat([1,6])
            out = torch.cat([img1_scores.unsqueeze(dim=1), img2_scores.unsqueeze(dim=1)], dim=1)
            # scores = torch.stack([img1_score, img2_score])
            # margin = scores.softmax(dim=0)[0] # Make comparison relative to first image
            # cl = self.assign_classification(margin)            
        else:
            cl = self.get_image_score(imgs, txt_feats)
            # if self.son_rescale:
            #     cl = (cl * 9) + 1

        if indices is not None:
            # cls = cls[range(cls.shape[0]), indices].flatten()
            out = out[range(out.shape[0]),:, indices]#.flatten()
            # img1_scores = img1_scores[range(img1_scores.shape[0]),:, indices].flatten()
            # img2_scores = img2_scores[range(img2_scores.shape[0]),:, indices].flatten()

        #  cls = self.apply_threshold(cls)
        return out

class ContrastiveManyPromptsNet(nn.Module):
    def __init__(self, clip_model, prompts, prompt_values):
        super().__init__()
        self.model = clip_model
        self.prompts = prompts
        self.prompt_values = prompt_values

    def simple_contrastive(self, img_feats, txt_feats):
        img_feats /= img_feats.norm(dim=-1, keepdim=True)
        txt_feats /= txt_feats.norm(dim=-1, keepdim=True)
        activations = (100.0 * img_feats @ txt_feats.T).softmax(dim=-1)
        output = activations * self.prompt_values.unsqueeze(0)
        return output.mean(-1) # Figure out case better

    def forward(self, img):
        img_feats = img 
        txt_feats = self.model.encode_text(self.prompts)        
        scenicness = self.simple_contrastive(img_feats, txt_feats)
        return scenicness

class ContrastivePromptsNet(nn.Module):
    def __init__(self, clip_model, prompts=None):
        super().__init__()
        self.model = clip_model
        self.prompts = prompts

    def simple_contrastive(self, img_feats, txt_feats):
        img_feats /= img_feats.norm(dim=-1, keepdim=True)
        txt_feats /= txt_feats.norm(dim=-1, keepdim=True)
        ranking = (100.0 * img_feats @ txt_feats.T).softmax(dim=-1)
        return ranking[:,0]

    def forward(self, img):
        img_feats = img 
        txt_feats = self.model.encode_text(self.prompts)        
        scenicness = self.simple_contrastive(img_feats, txt_feats)
        return scenicness

class PP2CLIPNet(nn.Module):
    def __init__(self, clip_model, prompts=None):
        super().__init__()
        self.model = clip_model
        self.prompts = prompts

    def simple_contrastive(self, img_feats, txt_feats):
        img_feats /= img_feats.norm(dim=-1, keepdim=True)
        txt_feats /= txt_feats.norm(dim=-1, keepdim=True)
        ranking = (100.0 * img_feats @ txt_feats.T).softmax(dim=-1)
        return ranking[:,0]

    def forward(self, img):
        img_feats = self.model.encode_image(img)
        scores = []
        for score_prompts in self.prompts:
            score_responses = []
            txt_feats = self.model.encode_text(score_prompts)
            score_responses = self.simple_contrastive(img_feats, txt_feats)
            scores.append(score_responses)
        scores = torch.stack(scores)
        return scores

class CLIPZeroShotModel(pl.LightningModule):
    def __init__(self, scatter_dir, net, run_name, label_info, splits):
        super().__init__()
        self.net = net
        self.label_info = label_info

        self.run_name = run_name
        self.out_dir = scatter_dir

        if 'train' in splits:
            self.train_tracker = VarTrackerCLIPExperiments(self.out_dir, 'train', label_info)
        if 'val' in splits:
            self.val_tracker = VarTrackerCLIPExperiments(self.out_dir, 'val', label_info)
        if 'test' in splits:
            self.test_tracker = VarTrackerCLIPExperiments(self.out_dir, 'test', label_info)

### General iteration functions ###
    def forward(self, x):
        x = self.net(x)
        return x

    def iteration_forward(self, batch, tracker, split):
        preds = (self.net(batch['img']) * 9) + 1

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
        return None

    def datapoints_to_tracker(self, tracker, datapts, var_name):
        length_is_one = len(datapts['preds']) == 1
        for attr in datapts:
            if length_is_one:
                datapts[attr] = [int(datapts[attr])]
            elif type(datapts[attr]) != list:
                datapts[attr] = datapts[attr].squeeze()
            
            # Store into tracker
            tracker.variables[var_name].attrs[attr].extend(datapts[attr])

    def end_epoch(self, tracker):
        ## Write outputs        
        tracker.store_epoch_metrics()
        tracker.print_results()
        tracker.save_metrics_to_file()
        tracker.save_observations_to_file(self.current_epoch)

        ## Reset for next epoch
        tracker.reset_epoch_vars()

### Testing ###
    def test_step(self, batch, batch_idx):
        loss = self.iteration_forward(batch, self.test_tracker, 'test')
        return loss

    def test_epoch_end(self, test_outputs):
        self.end_epoch(self.test_tracker)
        self.num_steps = 0

class CLIPZeroShotPP2(CLIPZeroShotModel):
    def __init__(self, scatter_dir, net, run_name, label_info, splits):
        super().__init__(scatter_dir, net, run_name, label_info, splits)

    def iteration_forward(self, batch, tracker): 
        preds = self.net(batch['img']) # Rescale to 1-10 to fit original ranges

        ## Get metadata
        ids = batch['ids'] # .cpu().numpy()
        lat = batch['lat'].cpu().numpy()
        lon = batch['lon'].cpu().numpy()
        preds_out = preds.detach().cpu().numpy()

        ## Create storage dicts
        datapts = {'ids':ids, 'lat':lat, 'lon':lon}

        for i,k in enumerate(list(self.label_info.keys())):
            score_dpts = deepcopy(datapts)
            score_dpts[f'preds'] = preds_out[i, :]
            score_dpts[f'gt'] = batch[k].detach().cpu().numpy()

            ## Store to trackers
            self.datapoints_to_tracker(tracker, score_dpts, k)
        return None

    def datapoints_to_tracker(self, tracker, datapts, var_name):
        length_is_one = len(datapts['preds']) == 1
        for attr in datapts:
            if length_is_one:
                datapts[attr] = [int(datapts[attr])]
                # ids = [int(ids)]         
            
            # Store into tracker
            tracker.variables[var_name].attrs[attr].extend(datapts[attr])

        # tracker.variables[var_name].vars['ids'].extend(ids)

### Validation ###
    def on_validation_epoch_start(self):
        ## Clear out val test run data ##
        if self.current_epoch == 0:
            self.val_tracker.reset_out_files()   

    def validation_step(self, batch, batch_idx):
        loss = self.iteration_forward(batch, self.val_tracker, 'val')
        return loss

    def validation_epoch_end(self, train_outputs):
        self.end_epoch(self.val_tracker)
        self.num_steps = 0

### Testing ###
    def test_step(self, batch, batch_idx):
        loss = self.iteration_forward(batch, self.test_tracker)
        return loss

    def test_epoch_end(self, test_outputs):
        self.end_epoch(self.test_tracker)
        self.num_steps = 0