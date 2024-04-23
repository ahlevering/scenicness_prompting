import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl
from codebase.experiment_tracking.run_tracker import VarTrackerCLIPExperiments

import torch.nn.functional as F

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

class PromptLearnerWrapper(nn.Module):
    def __init__(self, clip_model, coop_hyperparams):
        super().__init__()
        n_cls = len(coop_hyperparams['classes'])
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        ## Params        
        self.class_names = coop_hyperparams['classes']
        self.class_token_position = coop_hyperparams['class_pos'] # or middle
        self.n_ctx = coop_hyperparams['m_words']
        self.shots = coop_hyperparams['n_shots']
        self.m = coop_hyperparams['m_words']
        self.csc = coop_hyperparams['class_context']
        if 'ctx_init' in coop_hyperparams:
            self.ctx_init = coop_hyperparams['ctx_init']
        device_num = torch.get_device(next(iter(clip_model.visual.parameters())))

        if self.ctx_init:
            # use given words to initialize context vectors
            self.ctx_init = self.ctx_init.replace("_", " ")
            self.n_ctx = len(self.ctx_init.split(" "))
            prompt = clip.tokenize(self.ctx_init).to(device_num)
            with torch.no_grad():
                # In-place cast to CPU for prompt init
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + self.n_ctx, :]
            prompt_prefix = self.ctx_init

        else:
            # random initialization
            if self.csc:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, self.n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(self.n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * self.n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {self.n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)#.t()  # to be optimized

        classnames = [name.replace("_", " ") for name in self.class_names]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames] # Num tokens for each class
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device_num)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + self.n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        # self.n_ctx = n_ctx

        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts

class CLIPMultiPromptNet(nn.Module):
    def __init__(self, model, tokenized_prompts, prompt_values, use_embeddings=False, son_rescale=False):
        super().__init__()
        self.model = model
        self.encoded_prompts = self.model.encode_text(tokenized_prompts)
        self.prompt_values = prompt_values
        self.use_embeddings=use_embeddings
        self.son_rescale = son_rescale

    def simple_contrastive(self, img_feats, txt_feats):
        img_feats /= img_feats.norm(dim=-1, keepdim=True)
        txt_feats /= txt_feats.norm(dim=-1, keepdim=True)
        activations = (100.0 * img_feats @ txt_feats.T)
        # activations = (4.6052 * img_feats @ txt_feats.T)
        if self.son_rescale:
            prompt_activations = activations.softmax(dim=-1)
            output = prompt_activations * self.prompt_values.unsqueeze(0)
            # output = (prompt_activations * self.prompt_values.unsqueeze(0)* 0.9) + 1
            # output = (output.sum(-1) * 5) + 1
            # output = output.clamp(1, 10) # Zeroes are set to 1
            output = output.sum(dim=-1)
            # import pdb;pdb.set_trace()
        else:
            output = activations * self.prompt_values.unsqueeze(0)            
            output = output.mean(-1)
        return output

    def get_image_score(self, img, txt_feats):
        if self.use_embeddings:
            img_feats = img
        else:
            img_feats = self.model.visual(img)
        txt_feats = self.encoded_prompts
        scoring = self.simple_contrastive(img_feats, txt_feats)
        return scoring.unsqueeze(-1)

    def forward(self, imgs, indices=None):
        txt_feats = self.encoded_prompts            
        if type(imgs) == dict:
            img1_scores = self.get_image_score(imgs['img_left'], txt_feats).repeat([1,6])
            img2_scores = self.get_image_score(imgs['img_right'], txt_feats).repeat([1,6])
            out = torch.cat([img1_scores.unsqueeze(dim=1), img2_scores.unsqueeze(dim=1)], dim=1)
        else:
            out = self.get_image_score(imgs, txt_feats)
            # out = out.mean()
            if self.son_rescale:
                pass # out = (out * 9) + 1

        if indices is not None:
            out = out[range(out.shape[0]),:, indices]#.flatten()
        return out

class SONLinearProbe(nn.Module):
    def __init__(self, net, use_embeddings=False):
        super().__init__()
        self.baseline_net = net
        in_dims = 768
        self.fc = nn.Linear(in_dims, 1, bias=False)
        # with torch.no_grad():
        #     self.fc.bias.fill_(4.5)
        # self.model.apply(self.deactivate_batchnorm)
        self.use_embeddings = use_embeddings        

    def deactivate_batchnorm(self, m):
        if isinstance(m, nn.BatchNorm2d):
            m.reset_parameters()
            m.eval()
            with torch.no_grad():
                m.weight.fill_(1.0)
                m.bias.zero_()  

    def forward(self, image):
        if self.use_embeddings:
            features = image
        else:
            features = self.baseline_net(image.half())
        scenicness = self.fc(features.float()) + 4.43 # Fixed bias
        return scenicness

class CLIPMulticlassFewShotNet(nn.Module):
    def __init__(self, clip_model, coop_hyperparams):
        super().__init__()
        self.image_encoder = clip_model.visual
        self.coop_learner = CoOpCLIPLearner(clip_model, coop_hyperparams)
        self.use_embeddings = False

    def forward(self, imgs, indices):
        if not self.use_embeddings:        
            img_feats_left = self.image_encoder(imgs['img_left'].half())
            img_feats_right = self.image_encoder(imgs['img_right'].half())
        else:
            img_feats_left = imgs['img_left']
            img_feats_right = imgs['img_right']
        logits_left = self.coop_learner(img_feats_left)
        logits_right = self.coop_learner(img_feats_right)

        # Reshape to only class indices
        indices_double = torch.stack([indices*2, (indices*2)+1], dim=1)
        logits_left_for_class = logits_left[torch.arange(logits_left.shape[0]).unsqueeze(1), indices_double]
        logits_right_for_class = logits_right[torch.arange(logits_right.shape[0]).unsqueeze(1), indices_double]
        activations_left = logits_left_for_class.softmax(dim=1)[:,0] # Keep only positive class activations
        activations_right = logits_right_for_class.softmax(dim=1)[:,0]
        out = torch.cat([activations_left.unsqueeze(dim=1), activations_right.unsqueeze(dim=1)], dim=1)
        return out     

class CLIPFewShotModule(pl.LightningModule):
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
        self.log(f'{split}_mse', loss.detach().cpu(), on_step=False, on_epoch=True)        
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

    def on_save_checkpoint(self, checkpoint):
        # Keep only prompt head
        state = {}
        for key in checkpoint['state_dict']:
            if "net.coop_learner.prompt_learner" in key:
                state[key] = checkpoint['state_dict'][key]
            elif "baseline_full" in self.run_name:
                state[key] = checkpoint['state_dict'][key]
            elif not "unfrozen" in self.run_name:
                if "net.fc" in key:
                    state[key] = checkpoint['state_dict'][key]
            elif "score_regression" in self.run_name:
                if "prompt_values" in key:
                    state[key] = checkpoint['state_dict'][key]
        checkpoint['state'] = state
        del checkpoint['state_dict']

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

    def training_epoch_end(self, train_outputs):
        self.end_epoch(self.train_tracker)
        if 'scenic' in self.train_tracker.variables:
            self.log('train_r2', self.train_tracker.variables['scenic'].metrics['rsquared'][-1])        
        # if self.current_epoch == 0:
        #     for param in self.net.probe.parameters():
        #         param.requires_grad = True
        #     for param in self.net.coop_learner.csc.parameters():
        #         param.requires_grad = True            
            
            
# ### Validation ###
    def on_validation_epoch_start(self):
        self.net.use_embeddings = self.embeddings_policy['val']
        ## Clear out val test run data from quick dev runs ##
        if self.current_epoch == 0:
            self.val_tracker.reset_out_files()   

    def validation_step(self, batch, batch_idx):
        loss = self.iteration_forward(batch, self.val_tracker, 'val')
        return loss

    def validation_epoch_end(self, train_outputs):
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

    def test_epoch_end(self, test_outputs):
        self.end_epoch(self.test_tracker, store_outputs=False)
    
    def set_hyperparams(self, lr=0.0001, decay=0.0001):
        self.lr = lr
        self.decay = decay

    def configure_optimizers(self):
        if "baseline" in self.run_name:        
            # optimizer = torch.optim.SGD(self.net.parameters(), lr=self.lr, weight_decay=self.decay) #, momentum=False)
            optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.decay)
        else:
            optimizer = torch.optim.SGD(self.net.parameters(), lr=self.lr, weight_decay=self.decay) #, momentum=False)
            # optimizer = torch.optim.SGD(
            #     [
            #         {"params": self.net.text_scaler, "lr": self.lr*10000},
            #     ],
            #     lr=self.lr*10
            # )
        scheduler1 = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)

        return [optimizer], [scheduler1]

class CLIPMultiClassModule(CLIPFewShotModule):
    def __init__(self, scatter_dir, net, run_name, label_info, embeddings_policy, splits, tracker_class, lut=None):
        super().__init__(scatter_dir, net, run_name, label_info, embeddings_policy, splits, tracker_class)
        self.net = net
        self.label_info = label_info

        self.run_name = run_name
        self.out_dir = scatter_dir
        self.embeddings_policy = embeddings_policy

        ## Refactor later, class already initialized in parent class
        if 'train' in splits:
            self.train_tracker = VarTrackerCLIPExperiments(self.out_dir, 'train', label_info, tracker_class, lut)
        if 'val' in splits:
            self.val_tracker = VarTrackerCLIPExperiments(self.out_dir, 'val', label_info, tracker_class, lut)
        if 'test' in splits or 'all' in splits:
            self.test_tracker = VarTrackerCLIPExperiments(self.out_dir, 'test', label_info, tracker_class, lut)

    def iteration_forward(self, batch, tracker, split):                
        preds = self.net(batch['img'])
        gt = torch.stack(batch['gt']).T.double().squeeze()
        # loss = F.binary_cross_entropy(preds.double().squeeze(), gt)
        loss = 0

        ## Get metadata
        ids = batch['ids'].cpu().numpy()
        lat = batch['lat'].cpu().numpy()
        lon = batch['lon'].cpu().numpy()
        preds_out = preds.detach().cpu().numpy()
        gt_out = gt.detach().cpu().numpy()

        ## Create storage dicts
        datapts = {'ids':ids, 'lat':lat, 'lon':lon, 'preds':preds_out, 'gt':gt_out}

        ## Store to trackers
        self.datapoints_to_tracker(tracker, datapts, 'corine')
        # self.log(f'{split}_bce', loss.detach().cpu(), on_step=False, on_epoch=True)        
        return loss

    def training_epoch_end(self, train_outputs):
        self.end_epoch(self.train_tracker)
        self.log('train_f1', self.val_tracker.variables['corine'].metrics['f1'][-1])

    def validation_epoch_end(self, train_outputs):
        self.end_epoch(self.val_tracker)
        self.log('val_f1', self.val_tracker.variables['corine'].metrics['f1'][-1])

class CLIPComparisonModule(CLIPFewShotModule):
    def __init__(self, scatter_dir, net, run_name, label_info, embeddings_policy, splits, tracker_class):
        super().__init__(scatter_dir, net, run_name, label_info, embeddings_policy, splits, tracker_class)

### General iteration functions ###
    def forward(self, x):
        x = self.net(x)
        return x

    def iteration_forward(self, batch, tracker, split):                
        preds = self.net(batch['img'], batch['cat_index'])
        
        eq_batch_indices = (batch['cat_score'] == 0).nonzero(as_tuple=True)[0]
        # Calculate equal rankings
        if len(eq_batch_indices) > 0:
            eq_loss = (preds[:, 0] - preds[:, 1])[eq_batch_indices].abs().mean()
        else:
            eq_loss = 0

        indices_to_keep = [i for i in range(len(preds)) if i not in eq_batch_indices]
        margin_preds = preds[indices_to_keep, :]
        margin_gt = batch['cat_score'][indices_to_keep]
        margin_loss_func = nn.MarginRankingLoss(margin=0.01)
        margin_loss = margin_loss_func(margin_preds[:, 0].double().squeeze(),
                                       margin_preds[:, 1].double().squeeze(),
                                       margin_gt.squeeze().double()
                                       )
        loss = eq_loss + margin_loss

        ## Get metadata
        ids = batch['point_id_left']
        lat = batch['lat'].cpu().numpy()
        lon = batch['lon'].cpu().numpy()
        preds_out = preds.detach().cpu()
        max_indices = preds_out.argmax(dim=1)

        # Map to 1 (left image pref) or -1 (right image pref)
        classification_scores = np.zeros_like(max_indices, dtype=float)
        classification_scores[max_indices == 0] = 1.0
        classification_scores[max_indices == 1] = -1.0
        # classification_scores = classification_scores[indices_to_keep] # Keep only pair comparisons

        # Margin of 0.05 = images are equal
        diff = np.abs(preds_out[:, 0] - preds_out[:, 1])
        diff_mask = np.zeros_like(diff, dtype=bool)
        # diff_mask[eq_batch_indices.cpu()] = (diff[eq_batch_indices] < 0.05)
        # classification_scores[diff_mask] = 0

        gt_out = batch['cat_score'].detach().cpu().numpy()
        # gt_out = gt_out[indices_to_keep] # Keep only pair comparisons

        ## Create storage dicts
        datapts = {'ids':ids, 'lat':lat, 'lon':lon, 'preds':classification_scores, 'gt':gt_out, 'cat_name': batch['cat_name']}

        ## Store to trackers
        vars = ["lively", "depressing" , "boring", "beautiful", "safety", "wealthy"]
        for var in vars:
            var_indices = [i for i, c in enumerate(datapts['cat_name']) if c == var and i in indices_to_keep]
            var_batches = {}
            for key in datapts:
                var_batches[key] = [datapts[key][i] for i in var_indices]
            del var_batches['cat_name']
            if not len(var_batches['gt']) == 0:
                self.datapoints_to_tracker(tracker, var_batches, var)
        self.log(f'{split}_loss', loss.detach().cpu(), on_step=False, on_epoch=True, batch_size=len(preds))
        return loss

    def datapoints_to_tracker(self, tracker, datapts, var_name):
        length_is_one = len(datapts['preds']) == 1
        for attr in datapts:
            if length_is_one:
                pass #datapts[attr] = datapts[attr]
                # ids = [int(ids)]         
            elif type(datapts[attr]) != list:
                datapts[attr] = datapts[attr].squeeze()
            
            # Store into tracker
            tracker.variables[var_name].attrs[attr].append(datapts[attr])          