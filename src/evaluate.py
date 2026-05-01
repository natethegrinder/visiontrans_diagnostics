import numpy as np

# Order matches NIH_CHEST_XRAY_LABELS in data.py
DISEASE_LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
    'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
    'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia',
]


def binary_roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute binary ROC-AUC using average ranks, including tied scores."""
    y_true = np.asarray(y_true).astype(bool)
    y_score = np.asarray(y_score)
    num_pos = int(y_true.sum())
    num_neg = int((~y_true).sum())
    if num_pos == 0 or num_neg == 0:
        return float("nan")

    order = np.argsort(y_score, kind="mergesort")
    sorted_scores = y_score[order]
    ranks = np.empty(len(y_score), dtype=float)

    start = 0
    while start < len(y_score):
        end = start + 1
        while end < len(y_score) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end

    pos_rank_sum = ranks[y_true].sum()
    return float((pos_rank_sum - num_pos * (num_pos + 1) / 2.0) / (num_pos * num_neg))


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
        if y_true[:, i].sum() == 0 or y_true[:, i].sum() == y_true.shape[0]:
            results[label] = float('nan')
            continue
        auc = binary_roc_auc_score(y_true[:, i], y_pred[:, i])
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
