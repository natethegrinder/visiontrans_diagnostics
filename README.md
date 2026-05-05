# Objective:
- exp 1: cnn baseline final result: train/test outputs and mlflow results
- exp 5: bce vs focal result comparison with same config
- exp 3: cnn cam / grad-cam visiualization, loU accessment, bbox interprety

# exp 1
# resnet model BCE train
- python -m src.resnet_train --config configs/cnn_baseline.yaml --loss bce --device cuda
- best checkpoint will be saved to: outputs/checkpoints/cnn_resnet18_nih_baseline_bce.pt
- mlflow will record the full training process for this baseline run

# resnet model BCE evaluation 
- python -m src.evaluate --config configs/cnn_baseline.yaml --checkpoint outputs/checkpoints/cnn_resnet18_nih_baseline_bce.pt --loss bce --split test
- this step writes the NIH multilabel test metrics JSON used by later comparison/analysis

# exp 1 outputs:
- outputs/checkpoints/cnn_resnet18_nih_baseline_bce.pt, outputs/evaluations/cnn_resnet18_nih_baseline_bce_test.json
- mlflow run artifacts and metrics in the local sqlite backend `mlflow.db`

# exp 5
# resnet model Focal train
- python -m src.resnet_train --config configs/cnn_baseline.yaml --loss focal --device cuda
- best checkpoint will be saved to: outputs/checkpoints/cnn_resnet18_nih_baseline_focal.pt
- this keeps the same config as exp 1 and only changes the loss from BCE to focal

# resnet model Focal evaluation 
- python -m src.evaluate --config configs/cnn_baseline.yaml --checkpoint outputs/checkpoints/cnn_resnet18_nih_baseline_focal.pt --loss focal --split test
- this writes the focal test metrics JSON required by the comparison script

# bce vs focal comparison
- python -m src.compare_losses
- outputs: outputs/comparisons/bce_vs_focal_comparison.json and outputs/comparisons/bce_vs_focal_comparison.md
- by default this script compares `cnn_resnet18_nih_baseline_bce_test.json` and `cnn_resnet18_nih_baseline_focal_test.json`
- if your older BCE result is still named `cnn_resnet18_nih_baseline_test.json`, the script also accepts it automatically
- the comparison summary is also logged to mlflow

# exp 5 outputs:
- outputs/evaluations/cnn_resnet18_nih_baseline_focal_test.json, outputs/comparisons/bce_vs_focal_comparison.json

# exp 3
# cnn interpretability
- python -m src.interpretability --config configs/interpretability.yaml --checkpoint outputs/checkpoints/cnn_resnet18_nih_baseline_bce.pt --device cuda --limit 100
- outputs: outputs/interpretability/<checkpoint>_grad_cam_iou.json and outputs/interpretability/<checkpoint>/ overlay images
- this script runs Grad-CAM on the CNN checkpoint, resizes the heatmap back to the NIH original image size, and computes IoU against bbox annotations
- `--limit 100` means evaluating 100 bbox image-label pairs; you can increase or decrease it as needed

# exp 3 outputs: 
- outputs/interpretability/cnn_resnet18_nih_baseline_bce_grad_cam_iou.json, overlay images

# tuning footprints (did not run)
- python -m src.tune_thresholds --config configs/cnn_baseline.yaml --checkpoint outputs/checkpoints/cnn_resnet18_nih_baseline_bce.pt --loss bce --device cuda --tune-split val --report-split test
 
# results summarize
- 
- single-run report summary:
- python -m src.summarize_results --loss bce --split test
- python -m src.summarize_results --loss focal --split test
- each run writes one report JSON plus two figures under outputs/reports/: report_summary.json, f1_vs_epoch.png, confusion_matrix.png

- all-loss report summary and comparison:
- python -m src.summarize_results --all-losses --split test
- this keeps the existing single-loss outputs and additionally writes a combined comparison JSON and two comparison plots
- combined comparison outputs:
- outputs/reports/bce_vs_focal_test_report_summary.json
- outputs/reports/bce_vs_focal_test_report_metrics.png
- outputs/reports/bce_vs_focal_test_f1_vs_epoch.png
- in the same run, BCE and focal per-run report files are also generated automatically if the corresponding MLflow runs and evaluation JSON files already exist


