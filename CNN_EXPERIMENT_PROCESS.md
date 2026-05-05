# CNN Experiment Process

This document summarizes the CNN experiment design used in the `CNN-Full-Experiment` branch. The corresponding experiment notebook is `notebooks/cnn_train_eval_experiments.ipynb`.

## Objective

The CNN experiments establish a complete baseline suite for comparison against the ViT/VIP models. The internal CNN comparison focuses on two axes:

1. Architecture: ResNet18 vs DenseNet121.
2. Training setup: weighted BCE vs focal loss, and DenseNet121 with or without ImageNet pretraining.

The recommended primary CNN baseline is DenseNet121 pretrained with weighted BCE because it is the closest setup to CheXNet: DenseNet121, ImageNet pretraining, multi-label chest X-ray classification, and AUROC-focused model selection.

## Experiment Matrix

| Experiment | Config | Model | Pretraining | Loss | Purpose |
| --- | --- | --- | --- | --- | --- |
| ResNet18 BCE | `configs/cnn_resnet18_bce.yaml` | ResNet18 | false | weighted BCE | ResNet18 CNN baseline from random initialization |
| ResNet18 Focal | `configs/cnn_resnet18_focal.yaml` | ResNet18 | false | focal loss | ResNet18 loss ablation from random initialization |
| DenseNet121 No Pretrain BCE | `configs/cnn_densenet121_no_pretrain_bce.yaml` | DenseNet121 | false | weighted BCE | DenseNet121 pretraining ablation |
| DenseNet121 Pretrained BCE | `configs/cnn_densenet121_pretrained_bce.yaml` | DenseNet121 | true | weighted BCE | Main CheXNet-style CNN baseline |
| DenseNet121 Pretrained Focal | `configs/cnn_densenet121_pretrained_focal.yaml` | DenseNet121 | true | focal loss | DenseNet121 loss ablation |

## Shared Training Setup

All full CNN runs use the same data split and evaluation pipeline:

- Dataset: NIH Chest X-ray 14-label multi-label classification
- Input size: `224 x 224`
- Input channels: `1`
- Number of labels: `14`
- Optimizer: AdamW
- Learning rate: `1e-4`
- Weight decay: `1e-4`
- Scheduler: cosine annealing
- Mixed precision: enabled
- Batch size: `96` on AutoDL 4090
- Early stopping: validation mean AUROC with patience `7`
- Checkpoint selection: best validation mean AUROC

Weighted BCE refers to `BCEWithLogitsLoss(pos_weight=...)`. The positive class weights are computed from the training split and clipped by `max_pos_weight` to reduce instability from extremely rare labels.

## Metrics and Logging

The training script records the report-aligned metrics:

- Mean AUROC
- Mean PR AUC / mean average precision
- Macro F1
- Micro F1
- Macro precision
- Macro recall
- Per-label AUROC
- Per-label average precision
- Train loss and validation loss
- Training time
- Test time
- Peak GPU memory
- Average GPU memory
- Average GPU utilization
- Confusion matrix
- F1 score vs epoch

MLflow logging is enabled unless `--no-mlflow` is passed. The default tracking configuration is defined in `configs/base.yaml`.

## Outputs

Each experiment writes metrics under `artifacts/metrics/`:

```text
<run_name>_history.csv
<run_name>_val_auc.json
<run_name>_criterion.json
<run_name>_test_metrics.json
<run_name>_test_per_class_metrics.csv
<run_name>_loss_curve.png
<run_name>_f1_vs_epoch.png
<run_name>_test_confusion_matrix.png
```

Best checkpoints are written under `artifacts/models/`:

```text
<run_name>_best.pt
```

The report visualization section at the end of `notebooks/cnn_train_eval_experiments.ipynb` writes aggregate figures to:

```text
artifacts/metrics/report_figures/
```

## Report Visualizations

The notebook automatically builds:

- Validation mean AUROC convergence curves
- Train/validation loss and F1 convergence plots
- Final test metric comparison charts
- Precision-recall tradeoff scatter plot
- Per-label AUROC heatmap
- Per-label average precision heatmap
- Training budget comparison for time, peak GPU memory, and average GPU utilization
- Early stopping and checkpoint summary table

## Interpretability

CNN interpretability uses Grad-CAM with the trained DenseNet121 checkpoint:

```text
notebooks/cnn_densenet121_gradcam_outputs.ipynb
notebooks/cnn_densenet121_vit_aligned_interpretability.ipynb
```

These outputs can be compared with transformer attention maps to discuss local CNN responses versus more global transformer attention patterns.

Run `notebooks/cnn_train_eval_experiments.ipynb` first on AutoDL/CUDA or MacBook/MPS so that `artifacts/models/densenet121_pretrained_bce_best.pt` exists. Then run the two DenseNet121 interpretability notebooks.
