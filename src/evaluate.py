import numpy as np
from sklearn.metrics import roc_auc_score

# Order matches NIH_CHEST_XRAY_LABELS in data.py
DISEASE_LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
    'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
    'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia',
]


def compute_mean_auc(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    Args:
        y_true: (N, 14) binary ground-truth labels
        y_pred: (N, 14) predicted probabilities (post-sigmoid)
    Returns:
        dict: {label: auc, ..., 'mean': mean_auc}
        Classes with no positive samples get nan and are excluded from the mean.
    """
    results = {}
    aucs = []
    for i, label in enumerate(DISEASE_LABELS):
        if y_true[:, i].sum() == 0:
            results[label] = float('nan')
            continue
        auc = float(roc_auc_score(y_true[:, i], y_pred[:, i]))
        results[label] = auc
        aucs.append(auc)
    results['mean'] = float(np.mean(aucs)) if aucs else float('nan')
    return results


def print_auc_table(results: dict[str, float]) -> None:
    print(f"\n  {'Label':<22}  {'AUC':>6}")
    print(f"  {'-'*22}  {'-'*6}")
    for label in DISEASE_LABELS:
        val = results.get(label, float('nan'))
        marker = '  ← ' if val < 0.7 else ''
        print(f"  {label:<22}  {val:.4f}{marker}")
    print(f"  {'='*22}  {'='*6}")
    print(f"  {'Mean AUC':<22}  {results['mean']:.4f}")
