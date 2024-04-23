import yaml

##### LOAD SET-UP FILE #####
setup_file = "setup_files/test/son_zero_shot_multiprompt.yaml"
# setup_file = "setup_files/test/son_coop_contrastive_zero_shot.yaml"

with open(setup_file) as file:
    exp_params = yaml.full_load(file)

prompts = exp_params['hyperparams']['prompts']
values = list(prompts.values())
prompts = list(prompts.keys())
prompt_nums = (53, 119, 7, 36, 47, 51, 27, 43, 104, 89)
prompt_names = [prompts[p] for p in prompt_nums]
[print(f"{prompt_names[p]}: {values[p]}") for p, _ in enumerate(prompt_names)]