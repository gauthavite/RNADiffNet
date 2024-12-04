# RNADiffNet

This repository implements a custom version of **DiffusionNet**, inspired by the paper ["DiffusionNet: Discretization Agnostic Learning on Surfaces"](https://arxiv.org/abs/2206.09398).

## Dataset

We use the dataset introduced in the paper [Effective Rotation-invariant Point CNN with Spherical Harmonics kernels](https://arxiv.org/abs/1906.11555), which contains 640 RNA surface meshes of about 15k vertices each.

## Objective

The goal is to develop a model capable of accurately performing vertex segmentation of molecules using both their 3D meshes and point cloud representations.

## Methodology

We leverage **DiffusionNet**, a sampling- and resolution-agnostic model that operates directly on 3D meshes. It uses a learned diffusion layer to extract meaningful geometric features, enabling effective molecule segmentation.

## How to Run

1. Update the configuration parameters in `rna_config.py` as needed.
2. Run the following command to start training and testing:
   ```bash
   python main.py
   ```