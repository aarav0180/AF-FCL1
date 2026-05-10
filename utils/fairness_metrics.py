"""
Fairness / Bias evaluation metrics for federated continual learning.

Supports binary protected-group analysis with the following metrics:

  A. Demographic Parity    — Δ positive prediction rates between groups
  B. Equality of Opportunity — Δ True Positive Rates (TPR) between groups
  C. Equalized Odds         — Δ TPR + Δ FPR between groups
  D. Predictive Parity      — Δ Precision across groups

Also computes per-group confusion matrix, TPR, FPR, Precision, Recall.

All functions accept numpy arrays and are dataset-agnostic.

Usage:
    from utils.fairness_metrics import run_full_fairness_analysis, assign_groups_by_class
    groups = assign_groups_by_class(labels, strategy='even_odd')
    results = run_full_fairness_analysis(preds, labels, groups)
"""

import numpy as np
from collections import defaultdict


# ======================================================================
#  GROUP ASSIGNMENT
# ======================================================================

def assign_groups_by_class(labels: np.ndarray, strategy: str = 'even_odd') -> np.ndarray:
    """
    Assign binary protected group membership based on class labels.

    This is a proxy for real demographic groups — useful when no natural
    protected attribute exists (e.g. EMNIST, CIFAR-100).

    Parameters
    ----------
    labels : np.ndarray, shape [N]
        Ground-truth class labels.
    strategy : str
        'even_odd'   — even class IDs → group 0, odd → group 1
        'first_half' — classes < median → group 0, >= median → group 1

    Returns
    -------
    np.ndarray, shape [N], values in {0, 1}
    """
    if strategy == 'even_odd':
        # Even class labels → group 0, odd → group 1
        return (labels % 2).astype(int)
    elif strategy == 'first_half':
        # Split classes at the median
        median_class = np.median(np.unique(labels))
        return (labels >= median_class).astype(int)
    else:
        raise ValueError(f"Unknown group assignment strategy: {strategy}")


# ======================================================================
#  PER-GROUP CONFUSION MATRIX
# ======================================================================

def compute_confusion_matrix_per_group(preds: np.ndarray, labels: np.ndarray,
                                        groups: np.ndarray) -> dict:
    """
    Compute confusion-matrix-derived metrics for each group.

    For multi-class problems, we treat each class vs rest as a binary
    problem and macro-average.

    Parameters
    ----------
    preds  : np.ndarray, shape [N]
    labels : np.ndarray, shape [N]
    groups : np.ndarray, shape [N], values in {0, 1}

    Returns
    -------
    dict with keys 'group_0_metrics', 'group_1_metrics', each containing:
        accuracy, TPR, FPR, precision, recall, f1, num_samples
    """
    results = {}
    unique_classes = np.unique(labels)

    for g in [0, 1]:
        mask = (groups == g)
        if mask.sum() == 0:
            results[f'group_{g}_metrics'] = {
                'accuracy': 0.0, 'TPR': 0.0, 'FPR': 0.0,
                'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
                'num_samples': 0,
            }
            continue

        g_preds = preds[mask]
        g_labels = labels[mask]

        # Overall accuracy for this group
        accuracy = float(np.mean(g_preds == g_labels))

        # Macro-averaged binary metrics across all classes
        tpr_list, fpr_list, prec_list, recall_list = [], [], [], []

        for cls in unique_classes:
            # Binary: class cls vs rest
            tp = int(np.sum((g_preds == cls) & (g_labels == cls)))
            fp = int(np.sum((g_preds == cls) & (g_labels != cls)))
            fn = int(np.sum((g_preds != cls) & (g_labels == cls)))
            tn = int(np.sum((g_preds != cls) & (g_labels != cls)))

            # True Positive Rate (Sensitivity / Recall)
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            # False Positive Rate
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            # Precision
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            # Recall = TPR
            recall = tpr

            tpr_list.append(tpr)
            fpr_list.append(fpr)
            prec_list.append(prec)
            recall_list.append(recall)

        avg_tpr = float(np.mean(tpr_list))
        avg_fpr = float(np.mean(fpr_list))
        avg_prec = float(np.mean(prec_list))
        avg_recall = float(np.mean(recall_list))
        avg_f1 = (2 * avg_prec * avg_recall / (avg_prec + avg_recall)
                  if (avg_prec + avg_recall) > 0 else 0.0)

        results[f'group_{g}_metrics'] = {
            'accuracy': accuracy,
            'TPR': avg_tpr,
            'FPR': avg_fpr,
            'precision': avg_prec,
            'recall': avg_recall,
            'f1': avg_f1,
            'num_samples': int(mask.sum()),
        }

    return results


# ======================================================================
#  A. DEMOGRAPHIC PARITY
# ======================================================================

def compute_demographic_parity(preds: np.ndarray, groups: np.ndarray) -> float:
    """
    Demographic Parity: difference in positive prediction rates between groups.

    For multi-class: we compute the average per-class prediction rate
    difference across groups.

    DP = |P(ŷ=c | G=0) - P(ŷ=c | G=1)|  averaged over classes c.

    Returns
    -------
    float : absolute demographic parity difference (0 = perfect parity)
    """
    unique_classes = np.unique(preds)
    dp_diffs = []

    for cls in unique_classes:
        mask_g0 = (groups == 0)
        mask_g1 = (groups == 1)

        # Prediction rate for class cls in each group
        rate_g0 = np.mean(preds[mask_g0] == cls) if mask_g0.sum() > 0 else 0.0
        rate_g1 = np.mean(preds[mask_g1] == cls) if mask_g1.sum() > 0 else 0.0

        dp_diffs.append(abs(rate_g0 - rate_g1))

    return float(np.mean(dp_diffs))


# ======================================================================
#  B. EQUALITY OF OPPORTUNITY
# ======================================================================

def compute_equal_opportunity(preds: np.ndarray, labels: np.ndarray,
                               groups: np.ndarray) -> float:
    """
    Equality of Opportunity: difference in True Positive Rates (TPR)
    between groups, macro-averaged across classes.

    EO = |TPR_group0 - TPR_group1|  averaged over classes.

    Returns
    -------
    float : absolute EO difference (0 = perfect equality of opportunity)
    """
    unique_classes = np.unique(labels)
    eo_diffs = []

    for cls in unique_classes:
        tpr_per_group = []
        for g in [0, 1]:
            mask = (groups == g) & (labels == cls)
            if mask.sum() == 0:
                tpr_per_group.append(0.0)
            else:
                # TPR = P(ŷ=c | y=c, G=g)
                tpr = np.mean(preds[mask] == cls)
                tpr_per_group.append(float(tpr))

        eo_diffs.append(abs(tpr_per_group[0] - tpr_per_group[1]))

    return float(np.mean(eo_diffs))


# ======================================================================
#  C. EQUALIZED ODDS
# ======================================================================

def compute_equalized_odds(preds: np.ndarray, labels: np.ndarray,
                            groups: np.ndarray) -> dict:
    """
    Equalized Odds: differences in both TPR and FPR between groups.

    Returns
    -------
    dict with 'tpr_diff' and 'fpr_diff' (macro-averaged across classes)
    """
    unique_classes = np.unique(labels)
    tpr_diffs, fpr_diffs = [], []

    for cls in unique_classes:
        tpr_per_group = []
        fpr_per_group = []
        for g in [0, 1]:
            g_mask = (groups == g)
            # TPR: among true positives for this class in this group
            pos_mask = g_mask & (labels == cls)
            neg_mask = g_mask & (labels != cls)

            tpr = float(np.mean(preds[pos_mask] == cls)) if pos_mask.sum() > 0 else 0.0
            fpr = float(np.mean(preds[neg_mask] == cls)) if neg_mask.sum() > 0 else 0.0

            tpr_per_group.append(tpr)
            fpr_per_group.append(fpr)

        tpr_diffs.append(abs(tpr_per_group[0] - tpr_per_group[1]))
        fpr_diffs.append(abs(fpr_per_group[0] - fpr_per_group[1]))

    return {
        'tpr_diff': float(np.mean(tpr_diffs)),
        'fpr_diff': float(np.mean(fpr_diffs)),
    }


# ======================================================================
#  D. PREDICTIVE PARITY
# ======================================================================

def compute_predictive_parity(preds: np.ndarray, labels: np.ndarray,
                               groups: np.ndarray) -> float:
    """
    Predictive Parity: difference in Precision across groups,
    macro-averaged over classes.

    PP = |Precision_group0 - Precision_group1|  averaged over classes.

    Returns
    -------
    float : absolute predictive parity difference
    """
    unique_classes = np.unique(labels)
    pp_diffs = []

    for cls in unique_classes:
        prec_per_group = []
        for g in [0, 1]:
            g_mask = (groups == g)
            pred_pos = g_mask & (preds == cls)
            if pred_pos.sum() == 0:
                prec_per_group.append(0.0)
            else:
                # Precision = P(y=c | ŷ=c, G=g)
                prec = float(np.mean(labels[pred_pos] == cls))
                prec_per_group.append(prec)

        pp_diffs.append(abs(prec_per_group[0] - prec_per_group[1]))

    return float(np.mean(pp_diffs))


# ======================================================================
#  COMBINED ANALYSIS
# ======================================================================

def run_full_fairness_analysis(preds: np.ndarray, labels: np.ndarray,
                                groups: np.ndarray) -> dict:
    """
    Run all fairness metrics and return a comprehensive results dict.

    Parameters
    ----------
    preds  : np.ndarray, shape [N] — predicted class labels
    labels : np.ndarray, shape [N] — ground-truth class labels
    groups : np.ndarray, shape [N] — binary protected group (0 or 1)

    Returns
    -------
    dict : all metric values + per-group confusion-matrix stats
    """
    results = {}

    # A. Demographic Parity
    results['demographic_parity'] = compute_demographic_parity(preds, groups)

    # B. Equality of Opportunity
    results['equality_of_opportunity'] = compute_equal_opportunity(
        preds, labels, groups)

    # C. Equalized Odds
    eq_odds = compute_equalized_odds(preds, labels, groups)
    results['equalized_odds_tpr_diff'] = eq_odds['tpr_diff']
    results['equalized_odds_fpr_diff'] = eq_odds['fpr_diff']

    # D. Predictive Parity
    results['predictive_parity'] = compute_predictive_parity(
        preds, labels, groups)

    # Per-group confusion matrix stats
    group_metrics = compute_confusion_matrix_per_group(preds, labels, groups)
    results.update(group_metrics)

    # Group sizes
    results['group_0_size'] = int(np.sum(groups == 0))
    results['group_1_size'] = int(np.sum(groups == 1))
    results['overall_accuracy'] = float(np.mean(preds == labels))

    return results
