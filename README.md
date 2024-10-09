## About
This paper contains Scripts and data accompanying our publication titled *Prompt-guided and multimodal landscape scenicness
assessments with vision-language models*. We tested the use of vision-language models to provide scalable landscape assessments, as well as landscape assessments using only text as inputs. Please find the paper here: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0307083

## Example notebooks
* [Colab - making a personal map using rated landscape prompts](https://colab.research.google.com/drive/12vAQDB37Yk2GvFPb6lHF1Ws0a_PtqllV#scrollTo=4asBIqz_Pju3)
* Landscape Prompt Ensembling (coming soon)

## Updates
**30/09 - This repo will be updated soon to provide better boilerplate code for LPE, so that it is easier to re-use for new research.**
**09/10 - Added example notebook for making personalized maps. Will refactor LPE code and add a notebook example soon.**

### Data
Our dataset of prompts can be found in the following Zenodo repository: https://zenodo.org/records/12653736. Notably, the pre-computed embeddings can be downloaded from here. Alternatively, the ScenicOrNot image dataset can be constructed by running `preprocess/download_son_images.py`, although this will take a while, strains Geograph servers, and is not recommended if it can be avoided. A snapshot of the images will be provided here soon. For those interested in reproducing the exact results starting from the images (because some images have been taken offline), please feel free to contact the authors directly.

Auxillary data files such as split indices and the land cover class per image are also provided in this repository.

### How to use
We provide scripts for all of the methods in our paper.
- Baselines (`train/test_full_baseline.py`) - trains or evaluates a ViT-14L model, pretrained on either CLIP or SigLIP
- Few-shot learning (`few_shot_son_probes.py`) - Trains a linear model on pre-extracted embeddings in a few-shot manner
- Contrastive prompting (`contrastive_prompts.py`) - Evaluates contrastive prompt pairs specified in the setup .yaml files
- Landscape prompt ensembling (`landscape_prompt_ensembling.py`) - Performs two versions of prompt ensembling, namely early ensembling and late ensembling.

To use the scripts, install the required packages (`pip install -r requirements.txt`), download the embeddings from the Zenodo repo, and run the scripts. Most changes can be made in `/setup_files`.
