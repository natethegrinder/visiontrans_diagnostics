# VisionTrans Diagnostics

This repository contains the CNN and Vision Transformer training, evaluation, and interpretability workflow for multi-label chest X-ray diagnosis experiments.

The current CNN branch focuses on ResNet18 and DenseNet121 baselines for NIH Chest X-ray 14-label classification. The main DenseNet121 setup follows a CheXNet-style baseline: DenseNet121, ImageNet pretraining, weighted binary cross entropy, validation mean AUROC checkpoint selection, and final test-set evaluation.

## Repository Layout

```text
configs/      Experiment configuration files
notebooks/    Experiment tracking and interpretability notebooks
src/          Data loading, models, training, evaluation, and Grad-CAM utilities
artifacts/    Local model checkpoints, metrics, and generated figures
```

The data directory is expected to live outside the git repository, next to `visiontrans_diagnostics`:

```text
Group Project/
  data/
  visiontrans_diagnostics/
```

Default paths are configured in `configs/base.yaml`.

## CNN Config Hierarchy

CNN experiments use a layered YAML structure:

```text
base.yaml
  -> cnn_base.yaml
      -> cnn_resnet18_base.yaml
          -> cnn_resnet18_bce.yaml
          -> cnn_resnet18_focal.yaml
      -> cnn_densenet121_base.yaml
          -> cnn_densenet121_no_pretrain_bce.yaml
          -> cnn_densenet121_pretrained_bce.yaml
          -> cnn_densenet121_pretrained_focal.yaml
```

Shared data loader, optimizer, scheduler, mixed precision, early stopping, metric, artifact, and GPU logging settings are defined in `cnn_base.yaml`. Architecture-specific defaults are defined in the ResNet18 and DenseNet121 base configs. Each experiment config only overrides the run name, loss, or pretraining setting.

## Main CNN Experiments

Run the complete CNN experiment notebook:

```text
notebooks/cnn_train_eval_experiments.ipynb
```

It covers:

1. ResNet18 + weighted BCE
2. ResNet18 + focal loss
3. DenseNet121 + weighted BCE without pretraining
4. DenseNet121 + weighted BCE with ImageNet pretraining
5. DenseNet121 + focal loss with ImageNet pretraining

Full training logs parameters, metrics, checkpoints, and figures to MLflow unless `--no-mlflow` is passed.

## Notebook Execution Order

On an AutoDL Linux instance with an RTX 4090 and CUDA-enabled PyTorch, the notebooks select `cuda`. On a MacBook with a working PyTorch MPS backend, they select `mps`. Run the CNN notebooks in this order:

1. `notebooks/cnn_train_eval_experiments.ipynb`
   - Verifies the accelerator and data paths.
   - Runs the five CNN experiments when `RUN_FULL_SUITE = True`.
   - Produces metrics, MLflow logs, report figures, and the DenseNet121 checkpoint used by Grad-CAM.
2. `notebooks/cnn_densenet121_gradcam_outputs.ipynb`
   - Loads `artifacts/models/densenet121_pretrained_bce_best.pt`.
   - Generates DenseNet121 Grad-CAM examples, bbox IoU records, and per-label average heatmaps.
3. `notebooks/cnn_densenet121_vit_aligned_interpretability.ipynb`
   - Uses the same DenseNet121 checkpoint.
   - Builds a clean single-label index and runs the ViT-aligned CNN interpretability protocol.

## Interpretability

CNN Grad-CAM notebooks:

```text
notebooks/cnn_densenet121_gradcam_outputs.ipynb
notebooks/cnn_densenet121_vit_aligned_interpretability.ipynb
```

These notebooks load the trained DenseNet121 checkpoint and generate Grad-CAM overlays for qualitative comparison with transformer attention visualizations.
