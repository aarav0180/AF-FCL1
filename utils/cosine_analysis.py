"""
Cosine Similarity Analysis for AF-FCL replay credibility filtering.

Provides:
  - CosineTracker: accumulates per-batch cosine similarity statistics
    between flow-generated replay features and real data features
  - Utility functions for centroid-based and pairwise cosine analysis
  - Acceptance/rejection tracking based on cosine thresholds

The tracker is purely observational — it does NOT modify training logic.
It instruments the existing replay mechanism to capture cosine similarity
alongside the Gaussian-probability credibility weights already used.

Usage:
    tracker = CosineTracker(threshold=0.5)
    # Inside training loop:
    tracker.update(cos_sim_values)
    # After task ends:
    stats = tracker.get_task_summary()
    tracker.reset()
"""

import numpy as np
import torch
import torch.nn.functional as F


class CosineTracker:
    """
    Accumulates cosine similarity statistics across mini-batches within a task.

    Parameters
    ----------
    threshold : float
        Cosine similarity threshold for acceptance/rejection counting.
        Samples with cos_sim >= threshold are considered "accepted"
        (credible replay), below threshold are "rejected".
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()

    def reset(self):
        """Reset all accumulators for a new task."""
        self._all_sims = []      # list of floats (per-batch mean cos sim)
        self._all_raw = []       # list of 1-D arrays (all individual cos sims)
        self._num_accepted = 0
        self._num_rejected = 0
        self._total_samples = 0

    def update(self, cos_sims: np.ndarray):
        """
        Record a batch of cosine similarity values.

        Parameters
        ----------
        cos_sims : np.ndarray or float
            Cosine similarities for this batch. Can be a scalar (batch-mean
            cosine sim) or 1-D array of per-sample similarities.
        """
        if isinstance(cos_sims, (int, float)):
            cos_sims = np.array([cos_sims])
        elif isinstance(cos_sims, torch.Tensor):
            cos_sims = cos_sims.detach().cpu().numpy().flatten()
        else:
            cos_sims = np.asarray(cos_sims).flatten()

        self._all_sims.append(float(np.mean(cos_sims)))
        self._all_raw.append(cos_sims)

        # Acceptance/rejection counting
        accepted = int(np.sum(cos_sims >= self.threshold))
        rejected = int(np.sum(cos_sims < self.threshold))
        self._num_accepted += accepted
        self._num_rejected += rejected
        self._total_samples += len(cos_sims)

    def get_task_summary(self) -> dict:
        """
        Compute summary statistics for the current task.

        Returns
        -------
        dict with keys:
            mean, std, min, max — of all cosine similarities
            num_accepted, num_rejected, total — acceptance counts
            acceptance_rate — fraction of accepted samples
            per_batch_means — list of per-batch mean cosine sims
        """
        if self._total_samples == 0:
            return {
                'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0,
                'num_accepted': 0, 'num_rejected': 0, 'total': 0,
                'acceptance_rate': 0.0,
                'per_batch_means': [],
            }

        all_vals = np.concatenate(self._all_raw)
        acceptance_rate = (self._num_accepted / self._total_samples
                          if self._total_samples > 0 else 0.0)

        return {
            'mean': float(np.mean(all_vals)),
            'std': float(np.std(all_vals)),
            'min': float(np.min(all_vals)),
            'max': float(np.max(all_vals)),
            'num_accepted': self._num_accepted,
            'num_rejected': self._num_rejected,
            'total': self._total_samples,
            'acceptance_rate': acceptance_rate,
            'per_batch_means': [float(x) for x in self._all_sims],
        }

    def get_all_raw_values(self) -> np.ndarray:
        """Return all individual cosine similarity values as a flat array."""
        if len(self._all_raw) == 0:
            return np.array([])
        return np.concatenate(self._all_raw)


# ======================================================================
#  UTILITY FUNCTIONS
# ======================================================================

def compute_cosine_similarity_batch(features_a: torch.Tensor,
                                     features_b: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise cosine similarity between two feature batches.

    Parameters
    ----------
    features_a : torch.Tensor, shape [N, D]
    features_b : torch.Tensor, shape [M, D]

    Returns
    -------
    torch.Tensor, shape [N, M] — cosine similarity matrix
    """
    a_norm = F.normalize(features_a, p=2, dim=1)  # [N, D]
    b_norm = F.normalize(features_b, p=2, dim=1)  # [M, D]
    return torch.mm(a_norm, b_norm.t())             # [N, M]


def compute_centroid_cosine(flow_features: torch.Tensor,
                            real_features: torch.Tensor,
                            flow_labels: np.ndarray,
                            real_labels: np.ndarray) -> dict:
    """
    Compute cosine similarity between class centroids of flow-generated
    and real features.

    Parameters
    ----------
    flow_features : torch.Tensor, shape [N_flow, D]
    real_features : torch.Tensor, shape [N_real, D]
    flow_labels : np.ndarray, shape [N_flow]
    real_labels : np.ndarray, shape [N_real]

    Returns
    -------
    dict : class_label → cosine_similarity (between centroids)
    """
    result = {}
    common_classes = set(flow_labels.tolist()) & set(real_labels.tolist())

    for cls in common_classes:
        flow_mask = (flow_labels == cls)
        real_mask = (real_labels == cls)

        if flow_mask.sum() == 0 or real_mask.sum() == 0:
            continue

        flow_centroid = flow_features[flow_mask].mean(dim=0, keepdim=True)
        real_centroid = real_features[real_mask].mean(dim=0, keepdim=True)

        cos_sim = F.cosine_similarity(flow_centroid, real_centroid).item()
        result[int(cls)] = cos_sim

    return result


def compute_mean_cosine_sim(flow_features: torch.Tensor,
                            real_features: torch.Tensor) -> float:
    """
    Compute mean cosine similarity between centroids of flow-generated
    and real feature distributions (global, not per-class).

    Parameters
    ----------
    flow_features : torch.Tensor, shape [N, D]
    real_features : torch.Tensor, shape [M, D]

    Returns
    -------
    float : mean cosine similarity between distribution centroids
    """
    if flow_features.shape[0] == 0 or real_features.shape[0] == 0:
        return 0.0
    flow_centroid = flow_features.mean(dim=0, keepdim=True)
    real_centroid = real_features.mean(dim=0, keepdim=True)
    return float(F.cosine_similarity(flow_centroid, real_centroid).item())
