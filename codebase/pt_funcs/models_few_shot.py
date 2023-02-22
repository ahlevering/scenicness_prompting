import torch
from torch import nn
import pytorch_lightning as pl
from codebase.experiment_tracking.run_tracker import VarTrackerCLIPExperiments
import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

import torch.nn.functional as F

# from codebase.pt_funcs.models_zero_shot import 

_tokenizer = _Tokenizer()

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

#         clip_imsize = clip_model.visual.input_resolution
#         cfg_imsize = cfg.INPUT.SIZE[0]
#         assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

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
        ## TODO: COPY START AND END OF SEQUENCE TOKENS
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

class CLIPLinearProbe(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale

        self.probe = nn.Linear(512, 1)
        self.probe.bias.data.fill_(5)

    def forward(self, image):
        image = image.half() # Cast to half for compatibility
        img_features = self.image_encoder(image)
        scenicness = self.probe(img_features.float())
        return scenicness

class Baseline(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        x = self.model(image)
        return x

class CLIPLinearProbe(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale

        self.probe = nn.Linear(512, 1)
        self.probe.bias.data.fill_(5)

    def forward(self, image):
        image = image.half() # Cast to half for compatibility
        img_features = self.image_encoder(image)
        scenicness = self.probe(img_features.float())
        return scenicness

class CoOpCLIPLearner(nn.Module):
    def __init__(self, clip_model, coop_hyperparams):
        super().__init__()
        self.prompt_learner = PromptLearnerWrapper(clip_model, coop_hyperparams)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits

class SONCLIPFewShotNet(nn.Module):
    def __init__(self, basenet, coop_hyperparams):
        super().__init__()
        self.coop_learner = CoOpCLIPLearner(basenet, coop_hyperparams)

    # def simple_contrastive(self, img_feats, txt_feats):
    #     img_feats /= img_feats.norm(dim=-1, keepdim=True)
    #     txt_feats /= txt_feats.norm(dim=-1, keepdim=True)
    #     ranking = (100.0 * img_feats @ txt_feats.T).softmax(dim=-1)
    #     return ranking[:,0]

    def forward(self, img):
        # img_feats = self.model.encode_image(img)
        # txt_feats = self.model.encode_text(self.prompts)   
        logits = self.coop_learner(img)
        pos_likelihood = logits.softmax(dim=-1)[:,0]
        # scenicness = self.simple_contrastive(img_feats, txt_feats)
        return pos_likelihood # scenicness

# class PP2CLIPNet(nn.Module):
#     def __init__(self, clip_model, prompts=None):
#         super().__init__()
#         self.model = clip_model
#         self.prompts = prompts

#     def simple_contrastive(self, img_feats, txt_feats):
#         img_feats /= img_feats.norm(dim=-1, keepdim=True)
#         txt_feats /= txt_feats.norm(dim=-1, keepdim=True)
#         ranking = (100.0 * img_feats @ txt_feats.T).softmax(dim=-1)
#         return ranking[:,0]

#     def forward(self, img):
#         img_feats = self.model.encode_image(img)
#         scores = []
#         for score_prompts in self.prompts:
#             score_responses = []
#             txt_feats = self.model.encode_text(score_prompts)
#             score_responses = self.simple_contrastive(img_feats, txt_feats)
#             scores.append(score_responses)
#         scores = torch.stack(scores)
#         return scores

class CLIPFewShotModule(pl.LightningModule):
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
        if 'test' in splits or 'all' in splits:
            self.test_tracker = VarTrackerCLIPExperiments(self.out_dir, 'test', label_info)

### General iteration functions ###
    def forward(self, x):
        x = self.net(x)
        return x

    def iteration_forward(self, batch, tracker, split):                
        preds = (self.net(batch['img']) * 9) + 1 # Scale to 1-10 range

        loss = F.mse_loss(preds.double(), batch['gt'].double())

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
            if length_is_one:
                pass #datapts[attr] = datapts[attr]
                # ids = [int(ids)]         
            elif type(datapts[attr]) != list:
                datapts[attr] = datapts[attr].squeeze()
            
            # Store into tracker
            tracker.variables[var_name].attrs[attr].extend(datapts[attr])

        # tracker.variables[var_name].vars['ids'].extend(ids)

    def on_save_checkpoint(self, checkpoint):
        # Keep only prompt head
        prompt_state = {}
        for key in checkpoint['state_dict']:
            if "net.coop_learner.prompt_learner" in key:
                prompt_state[key] = checkpoint['state_dict'][key]
            elif "probe" in key:
                prompt_state[key] = checkpoint['state_dict'][key]
            elif "Baseline" in key:
                prompt_state[key] = checkpoint['state_dict'][key]
        checkpoint['prompter'] = prompt_state
        del checkpoint['state_dict']

    def end_epoch(self, tracker):
        tracker.store_epoch_metrics()
        
        ## Write outputs
        tracker.save_metrics_to_file()
        tracker.save_observations_to_file(self.current_epoch)
        # tracker.save_scatterplot(self.current_epoch)

        ## Reset for next epoch
        tracker.reset_epoch_vars()
        # tracker.print_results()

### Training ###
    def training_step(self, batch, batch_idx):
        loss = self.iteration_forward(batch, self.train_tracker, 'train')
        return loss

    def training_epoch_end(self, train_outputs):
        self.end_epoch(self.train_tracker)
        # if self.current_epoch == 0:
        #     for param in self.net.probe.parameters():
        #         param.requires_grad = True
        #     for param in self.net.coop_learner.csc.parameters():
        #         param.requires_grad = True            
            
            
# ### Validation ###
#     def on_validation_epoch_start(self):
#         ## Clear out val test run data ##
#         if self.current_epoch == 0:
#             self.val_tracker.reset_out_files()   

#     def validation_step(self, batch, batch_idx):
#         loss = self.iteration_forward(batch, self.val_tracker, 'val')
#         return loss

#     def validation_epoch_end(self, train_outputs):
#         self.end_epoch(self.val_tracker)
#         self.num_steps = 0

    # def optimizer_step(
    #     self,
    #     epoch,
    #     batch_idx,
    #     optimizer,
    #     optimizer_idx,
    #     optimizer_closure,
    #     on_tpu=False,
    #     using_lbfgs=False,
    #     using_native_amp=False,
    # ):
    #     # update params
    #     optimizer.step(closure=optimizer_closure)

    #     # skip the first 500 steps
    #     if self.trainer.global_step < 250:
    #         lr_scale = min(1.0, float(self.trainer.global_step + 1) / 250.0)
    #         for pg in optimizer.param_groups:
    #             pg["lr"] = lr_scale * self.lr

### Testing ###
    def test_step(self, batch, batch_idx):
        loss = self.iteration_forward(batch, self.test_tracker, 'test')
        return loss

    def test_epoch_end(self, test_outputs):
        self.end_epoch(self.test_tracker)
        self.num_steps = 0
    
    def set_hyperparams(self, lr=0.0001, decay=0.0001):
        self.lr = lr
        self.decay = decay

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.net.parameters(), lr=self.lr, weight_decay=self.decay)#, momentum=False)
        # optimizer = torch.optim.Adam(self.net.coop_learner.parameters(), lr=self.lr, weight_decay=self.decay)

        scheduler1 = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9, last_epoch=-1)

        return [optimizer], [scheduler1]