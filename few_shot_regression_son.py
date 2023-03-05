import yaml
from datetime import datetime
import warnings

import clip
import torch
import numpy as np
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.seed import seed_everything

from codebase.pt_funcs.dataloaders import SONData, ClipDataLoader
from codebase.pt_funcs.models_zero_shot import CLIPZeroShotModel, ManyPromptsNet
from codebase.experiment_tracking.save_metadata import ExperimentOrganizer

from sklearn import linear_model
from scipy.stats import kendalltau

##### SET GLOBAL OPTIONS ######
seed_everything(113)
np.set_printoptions(suppress=True)
torch.set_printoptions(sci_mode=False)

#### Shutting up annoying warnings ####
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

##### LOAD SET-UP FILE #####
setup_file = "setup_files/test/son_zero_shot.yaml"
with open(setup_file) as file:
    exp_params = yaml.full_load(file)

run_name = exp_params['descriptions']['name']
run_family = exp_params['descriptions']['exp_family']

basenet, preprocess = clip.load('ViT-B/32')
# model = model.to(device=exp_params['hyperparams']['gpu_num'])

##### SET UP TRANSFORMS #####
transforms = {'test': None}

label_info = {}
label_info['scenic'] = {}
label_info['scenic']['ylims'] = [1, 10]

##### SETUP LOADERS #####
data_module = ClipDataLoader( exp_params['hyperparams']['workers'],    
                              exp_params['hyperparams']['batch_size'],
                              data_class=SONData
                            )

data_module.setup_data_classes( exp_params['paths']['labels_file'],
                                exp_params['paths']['images_root'],
                                split_ids=None,
                                embeddings=exp_params['paths']['embeddings'],
                                transforms=transforms,
                                id_col=exp_params['descriptions']['id_col'],
                                splits=exp_params['descriptions']['splits']
                            )

loader_emulator = next(iter(data_module.test_data))

##### STORE ENVIRONMENT AND FILES #####
run_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
print(f"\nExperiment start: {run_time}\n")
base_path = f"runs/{run_family}/{run_name}/{run_time}/train/"
organizer = ExperimentOrganizer(base_path)
organizer.store_yaml(setup_file)
organizer.store_environment()
organizer.store_codebase(['.py'])

##### SETUP MODEL #####

all_data = ["A photo of a natural area.", 6,
                        "A photo of snow capped mountains", 10,
                        "A photo of a scenic lake.", 9,
                        "A photo of a scenic and snowy wonderland", 10,
                        "A photo of an idilic landscape", 9,
                        "A carpet of flowers in the forest", 9,
                        "A photo of a field.", 6,
                        "A photo of an unremarkable rural area.", 5,
                        "A photo of an urban area.", 4,
                        "A photo of a highway.", 1,
                        "A photo of a construction area.", 1,
                        "A photo of vehicles.", 1,
                        "Man made metal structures.", 1,
                        "An urban park", 5,
                        "Agricultural machinery", 2,
                        "A road between fields", 3]
all_data = ["A photo of a rolling countryside", 8,
 "A photo of a dramatic coastline", 9,
 "A photo of a tranquil river", 8,
 "A photo of a bustling market town", 7,
 "A photo of a misty forest", 8,
 "A photo of an abandoned industrial site", 4,
 "A photo of a quaint fishing village", 9,
 "A photo of a majestic castle", 10,
 "A photo of a desolate moorland", 6,
 "A photo of a picturesque village green", 8,
 "A photo of a futuristic cityscape", 6,
 "A photo of a serene beach", 9,
 "A photo of a verdant valley", 8,
 "A photo of a historic cathedral", 9,
 "A photo of a windswept hillside", 7,
 "A photo of a colourful garden", 8,
 "A photo of an imposing skyscraper", 5,
 "A photo of a quaint country pub", 7,
 "A photo of a pristine waterfall", 9,
 "A photo of a lively music festival", 6,
 "A photo of a charming thatched cottage", 8,
 "A photo of a rugged mountain range", 9,
 "A photo of a sleepy hamlet", 7,
 "A photo of a vibrant street market", 6,
 "A photo of a serene lake district", 9,
 "A photo of a gloomy urban alleyway", 3,
 "A photo of a quaint canal", 8,
 "A photo of a bustling train station", 6,
 "A photo of a wild and windswept beach", 8,
 "A photo of a quaint seaside resort", 7,
 "A photo of a tranquil rural churchyard", 8,
 "A photo of a lively city square", 7,
 "A photo of a rolling golf course", 6,
 "A photo of a lonely lighthouse", 8,
 "A photo of a majestic waterfall", 9,
 "A photo of a sleepy rural village", 7,
 "A photo of a bustling shopping centre", 5,
 "A photo of a serene riverside path", 8,
 "A photo of a peaceful country lane", 8,
 "A photo of a bustling city street", 6,
 "A photo of a desolate coastal plain", 6,
 "A photo of a quirky urban street art", 5,
 "A photo of a lively harbour", 7,
 "A photo of a picturesque thatched cottage", 8,
 "A photo of a tranquil Japanese garden", 9,
 "A photo of a majestic rolling hill", 8,
 "A photo of a gritty urban graffiti", 3,
 "A photo of a quaint rural windmill", 8,
 "A photo of a bustling farmer's market", 7,
 "A photo of a tranquil forest trail", 8,
 "A photo of a desolate urban wasteland", 3,
 "A photo of a charming cobbled street", 8,
 "A photo of a serene rural pond", 8,
 "A photo of a lively street festival", 6,
 "A photo of a majestic stately home", 9,
 "A photo of a peaceful woodland glade", 8,
 "A photo of a bustling city square", 7,
 "A photo of a wild and windswept moorland", 7,
 "A photo of a quaint rural bridge", 8,
 "A photo of a vibrant cultural festival", 6,
 "A photo of a tranquil rural meadow", 5]

contrastive_prompts = all_data[0:-2:2]
promps_values = all_data[1:-1:2]

prompts = torch.cat([clip.tokenize(p) for p in contrastive_prompts])
prompts = prompts.to(device=exp_params['hyperparams']['gpu_num'])


promps_values = torch.tensor(promps_values).to(device=exp_params['hyperparams']['gpu_num'])


net = ManyPromptsNet(basenet, prompts,promps_values)
model = CLIPZeroShotModel(organizer.root_path+'outputs/', net, run_name, label_info, ['test'])

dataloader =  torch.utils.data.DataLoader(data_module.test_data,batch_size=200,shuffle=True)

iterable_dataloader = iter(dataloader)

training_samples = next(iterable_dataloader)

x = model(training_samples['img'].cuda()).cpu().detach()
y = training_samples['gt']
reg = linear_model.Ridge(alpha=0.1).fit(x, y)

predictions = []
gts = []
for i in range(1000):
    samples = next(iterable_dataloader)
    with torch.no_grad():
        x = model(samples['img'].cuda()).cpu().detach().numpy()
    pred = reg.predict(x)
    predictions.append(pred)
    y = samples['gt'].numpy()
    gts.append(y)
gts = np.concatenate(gts)
predictions = np.concatenate(predictions)
tau, pvalue = kendalltau(predictions,gts)

##### SETUP TESTER #####
tb_logger = TensorBoardLogger(save_dir=organizer.logs_path, name=run_name)
trainer = Trainer(  gpus=[exp_params['hyperparams']['gpu_num']],
                    logger = tb_logger,
                    fast_dev_run=False,
                )

##### FIT MODEL #####
print(f"fitting {run_name}")
trainer.test(model, datamodule=data_module)