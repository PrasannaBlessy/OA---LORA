# OA-LoRA Training

This directory contains the training scripts used for the proposed
Occlusion-Aware Low-Rank Adaptation (OA-LoRA) framework.

## Training Pipeline

The training process consists of the following stages:

1. Prepare the TuSimple and CULane datasets using the scripts provided
   in the `datasets` directory.
2. Configure the training parameters using the YAML configuration files
   in the `configs` directory.
3. Initialize the SDXL 1.0 diffusion model.
4. Apply the proposed OA-LoRA adaptation strategy.
5. Train the model using the prepared lane image data.
6. Save the trained OA-LoRA weights for subsequent synthetic lane image
   generation.

## Main Training Script

The main training entry point is:

`train_oalora.py`

## Configuration

Training configurations are provided for:

- TuSimple: `configs/oalora_tusimple.yaml`
- CULane: `configs/oalora_culane.yaml`

## Hardware

The experiments were conducted using an NVIDIA RTX A6000 GPU with
48 GB of VRAM.

## Notes

The original TuSimple and CULane datasets are not redistributed in this
repository. Users should obtain the datasets from their respective
official or publicly available sources and follow the dataset
preparation instructions provided in the `datasets` directory.
