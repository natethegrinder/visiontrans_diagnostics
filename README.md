# Visiontrans Diagnostics
Comparative analysis of Convolutional Neural Networks (ResNet) and Vision Transformers for medical tumor classification using NIH and Lung CT datasets. This repository contains the code and experimental evidence for an iterative deep learning project, including model fine-tuning, performance comparison, and interpretability analysis. 
 
# Data Processing  

This document summarizes the data pipeline for the NIH Chest X-ray project, focusing on `data.py`, related configs, and how processed images are passed into the ViT model.

## 1. Goal

The data pipeline prepares raw NIH Chest X-ray images and labels for multilabel classification.

It handles:

- locating/exporting annotation files
- parsing disease labels
- building train/validation/test manifests
- applying preprocessing and augmentation
- computing class imbalance weights
- returning image tensors and multilabel targets for training

The ViT patching logic is handled inside the model, not inside `data.py`.

---

## 2. Dataset Inputs

Expected structure:

```text
data/
├── raw/
│   └── images/
├── annotations/
│   ├── Data_Entry_2017.csv
│   └── BBox_List_2017.csv
└── manifests/
    ├── train.csv
    ├── val.csv
    └── test.csv
```

## Overall Flow 

```
Raw NIH images and metadata
    ↓
Export/locate annotation files
    ↓
Parse multilabel disease targets
    ↓
Split by Patient ID
    ↓
Create train/val/test manifests
    ↓
Load image from manifest
    ↓
Apply resize, augmentation, normalization
    ↓
Return image tensor and target vector
    ↓
Batch with DataLoader
    ↓
Pass image batch to ViT model
    ↓
Patch embedding inside model
    ↓
Transformer encoder
    ↓
14 disease logits
```