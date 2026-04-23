### Project Contribution Matrix

| Task Module | Team Members | Key Deliverables | Rubric Alignment |
| :--- | :--- | :--- | :--- |
| **CNN Branch** | Yujun Zheng, JIAJIAN ZHU | `models/resnet_custom.py`, CNN training loop, hyperparameter tuning | Approach & Implementation (10 pts) |
| **ViT Branch** | YU ZHIGANG, Kai Li | `models/vit_custom.py`, ViT attention head tuning, LR warmup strategies | Approach & Implementation (10 pts) |
| **Data & Config** | JIAJIAN ZHU, Kai Li | `configs/`, `data_loader.py`, unified preprocessing, augmentation | Datasheets & Training (10 pts) |
| **Evaluation** | Yujun Zheng, YU ZHIGANG | MLflow experiment logs, ROC/AUC plots, performance analysis | Results & Analysis (10 pts) |


# Tasks Breakdown
---
## Project Architecture and Setup 
 - Project File Hierarchy 
 - Python Package and environment setup
 - Data pipeline and experiment runs observability setup  

## Data and Config 
### Data Preprocessing 
 - Data collection
 - Data preprocessing (transformation)
 - Data preparation 
   - output data that work for CNN and VIT respectively 
### Config 
 -  base config
    - shared defaults: data paths, image size, seed, optimizer defaults, logging, device
 - cnn baseline  
    - inherits base, set models (CNN/ResNet), baseline training settings 
 - vit baseline
    - inherits base, set models (VIT), VIT-specific patch size, depth, heads, warmup 
- robustness 
    - inherits base, loads trained checkpoints, applies corruptions, evaluates degradation 
- interpretability 
    - inherits base, loads trained checkpoints, runs Grad-CAM / attention rollout / BBox IOU

### Stakeholders and Responsibilities 
- Kai Li 
  - Data preprocessing for VIT 
