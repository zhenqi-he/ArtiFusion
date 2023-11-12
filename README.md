# Artifact Restoration in Histology Images with Diffusion Probabilistic Models

This is the official implement of **Artifact Restoration in Histology Images with Diffusion Probabilistic Models** (MICCAI2023) [Arxiv](https://arxiv.org/abs/2307.14262)

## Introduction

This is the first attempt at a denoising diffusion probabilistic model for histological artifact restoration, called ArtiFusion. Specifically, ArtiFusion formulates the artifact region restoration as a gradual denoising process, and its training relies solely on artifact-free images to simplify the training complexity. Furthermore, to capture local-global correlations in the regional artifact restoration, a novel Swin-Transformer denoising architecture is designed, along with a time token scheme. Our extensive evaluations demonstrate the effectiveness of ArtiFusion as a pre-processing method for histology analysis, which can successfully preserve the tissue structures and stain style in artifact-free regions during the restoration.

## Dataset
The dataset is a subset of Camelyon17. You may download from the following link.
[here](https://drive.google.com/drive/folders/1R75R2WjEpZMZU3l2GKPzsNpCmIBJN-ps?usp=drive_link)

## Train

The proposed ArtiFusion learns the capability of generating local tissue representation from contextual information during the training stage. We follow the training procedure in [guided-diffusion](https://github.com/openai/guided-diffusion)

Run [train.sh](model_training/train.sh) to train a DDPM model
```bash
sh train.sh
```
Run [sample.sh](model_training/sample.sh) for sampling from trained model
```bash
sh sample.sh
```

## Inference
```bash
python test.py --conf_path confs/XXXX.yml
```


## Acknowledgement
We develop our code based on the implementation of [RePaint](https://github.com/andreas128/RePaint) and [guided-diffusion](https://github.com/openai/guided-diffusion). And thanks to [Yiqing Shen](https://github.com/yiqings) for the contribution of the codes for down-sample classification tasks.
