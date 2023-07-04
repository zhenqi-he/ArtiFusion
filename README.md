# Artifact Restoration in Histology Images with Diffusion Probabilistic Models

This is the official implement of **Artifact Restoration in Histology Images with Diffusion Probabilistic Models** (MICCAI2023)

## Introduction

This is the first attempt at a denoising diffusion probabilistic model for histological artifact restoration, called ArtiFusion. Specifically, ArtiFusion formulates the artifact region restoration as a gradual denoising process, and its training relies solely on artifact-free images to simplify the training complexity. Furthermore, to capture local-global correlations in the regional artifact restoration, a novel Swin-Transformer denoising architecture is designed, along with a time token scheme. Our extensive evaluations demonstrate the effectiveness of ArtiFusion as a pre-processing method for histology analysis, which can successfully preserve the tissue structures and stain style in artifact-free regions during the restoration.

## Train

The proposed ArtiFusion learns the capability of generating local tissue representation from contextual information during the training stage. We follow the training procedure in [guided-diffusion](https://github.com/openai/guided-diffusion)
