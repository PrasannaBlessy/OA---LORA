# Dataset Preparation (
- TuSimple
- CULane)

This directory contains the dataset preparation scripts used for the OA-LoRA experiments.

The experiments use publicly available lane datasets:

- TuSimple
- CULane

The original datasets are not redistributed in this repository.
Users should obtain the datasets from their respective sources and
place them in the local directory structure described below.



# ACDC Nighttime Environmental Data Preparation

This folder provides the preprocessing code used to prepare nighttime environmental images from the publicly available ACDC dataset for OA-LoRA training.

## 1. Download ACDC

Download the original ACDC dataset from the official project website:

https://acdc.vision.ee.ethz.ch/

The original ACDC dataset is not redistributed in this repository.

## 2. Extract the Dataset

After downloading the dataset, extract the files locally.

Example:

D:/Datasets/ACDC/

## 3. Configure the Dataset Path

Open:

prepare_acdc_night.py

Update:

ACDC_ROOT = Path(r"D:\Datasets\ACDC")

according to your local ACDC dataset location.

## 4. Run the Preparation Script

Run:

python prepare_acdc_night.py

The script extracts the nighttime images and organizes them into training, validation, and test folders.

## 5. Output Structure

The processed dataset is organized as:

dataset/
└── environmental/
    └── acdc_night/
        ├── train/
        │   └── images/
        ├── val/
        │   └── images/
        └── test/
            └── images/

## 6. Dataset Usage

The processed nighttime images are used as environmental-condition references for OA-LoRA training. The ACDC images are not redistributed through this repository.
