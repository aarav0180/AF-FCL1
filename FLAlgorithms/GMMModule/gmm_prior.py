"""
TaskGMMPrior — differentiable Gaussian Mixture Model base distribution
for the normalizing flow in AF-FCL.

This is a drop-in replacement for nflows.distributions.normal.StandardNormal.
After each continual-learning task, call fit(z_numpy) to update the stored
cluster parameters (means, diagonal variances, mixing weights).

Before fitting (task 0), the prior falls back to N(0, I) so the flow can
bootstrap normally.

The loss function for the flow becomes:

    loss = -log p_GMM(z) - log|det J_f|
         = -flow.log_prob(xa, context=y_one_hot).mean()

where z = f(xa | y) is the latent code and p_GMM captures the K-component
structure of the PREVIOUS task's latent distribution.  This forces the
current task's feature manifold to stay semantically close to the clusters
learned in the last task — preventing semantic drift.

All log_prob computations are in pure PyTorch (no sklearn at inference time),
so gradients flow correctly through z during flow training.
"""

import numpy as np
import torch

from nflows.distributions.base import Distribution

_LOG_2PI = float(np.log(2.0 * np.pi))


class TaskGMMPrior(Distribution):
    """
    Diagonal-covariance Gaussian Mixture Model as a normalizing-flow
    base distribution.

    Parameters
    ----------
    feature_dim : int
        Dimensionality of the latent space (matches xa_dim, default 512).
    K : int
        Number of GMM components.
    """

    def __init__(self, feature_dim: int, K: int = 4):
        super().__init__()
        self.feature_dim = feature_dim
        self.K = K
        self._fitted = False

        # Buffers are moved automatically with .to(device) and are NOT
        # gradient parameters — the GMM is fixed after fitting.
        self.register_buffer('_means',       torch.zeros(K, feature_dim))
        self.register_buffer('_log_vars',    torch.zeros(K, feature_dim))   # log σ²
        self.register_buffer('_log_weights', torch.full((K,), -float(np.log(K))))  # uniform

    # ------------------------------------------------------------------
    # Fitting  (called once per task end, CPU side via sklearn)
    # ------------------------------------------------------------------

    def fit(self, z: np.ndarray) -> None:
        """
        Fit K diagonal-covariance Gaussian components to latent codes z.

        Parameters
        ----------
        z : np.ndarray, shape [N, D]
            Latent codes produced by running the full training set through
            flow._transform(xa | y_one_hot).  Collected by
            GMMPreciseModel.fit_gmm_prior().
        """
        from sklearn.mixture import GaussianMixture

        # Never request more components than we have samples
        K_eff = min(self.K, z.shape[0])

        gm = GaussianMixture(
            n_components=K_eff,
            covariance_type='diag',
            max_iter=200,
            n_init=3,
            random_state=42,
        )
        gm.fit(z)

        device = self._means.device
        means       = torch.tensor(gm.means_,
                                   dtype=torch.float32, device=device)          # [K_eff, D]
        log_vars    = torch.tensor(np.log(gm.covariances_ + 1e-8),
                                   dtype=torch.float32, device=device)          # [K_eff, D]
        log_weights = torch.tensor(np.log(gm.weights_ + 1e-30),
                                   dtype=torch.float32, device=device)          # [K_eff]

        # Pad to self.K if fewer samples than components (rare edge case)
        if K_eff < self.K:
            pad = self.K - K_eff
            means       = torch.cat([means,       torch.zeros(pad, self.feature_dim, device=device)], dim=0)
            log_vars    = torch.cat([log_vars,    torch.zeros(pad, self.feature_dim, device=device)], dim=0)
            log_weights = torch.cat([log_weights, torch.full((pad,), -1e9, device=device)], dim=0)

        self._means.copy_(means)
        self._log_vars.copy_(log_vars)
        self._log_weights.copy_(log_weights)
        self._fitted = True

    # ------------------------------------------------------------------
    # Distribution interface (nflows-compatible)
    # ------------------------------------------------------------------

    def _log_prob(self, z: torch.Tensor, context=None) -> torch.Tensor:
        """
        log p(z) = log Σ_k π_k N(z ; μ_k, diag(σ²_k))

        Parameters
        ----------
        z : torch.Tensor, shape [N, D]

        Returns
        -------
        torch.Tensor, shape [N]
        """
        if not self._fitted:
            # Task-0 fallback: standard normal N(0, I)
            return -0.5 * (z.pow(2) + _LOG_2PI).sum(dim=-1)

        D = self.feature_dim

        # Expand for broadcasting
        z_e = z.unsqueeze(1)             # [N, 1, D]
        mu  = self._means.unsqueeze(0)   # [1, K, D]
        lv  = self._log_vars.unsqueeze(0)  # [1, K, D]

        # log N(z ; μ_k, diag(σ²_k))
        #   = -0.5 * [D·log(2π) + Σ_d log(σ²_d) + Σ_d (z_d - μ_d)² / σ²_d]
        log_det  = lv.sum(dim=-1)                                 # [1, K]
        maha     = ((z_e - mu).pow(2) / lv.exp()).sum(dim=-1)     # [N, K]
        log_gauss = -0.5 * (D * _LOG_2PI + log_det + maha)       # [N, K]

        # log π_k + log N_k(z), then log-sum-exp over K
        log_components = self._log_weights.unsqueeze(0) + log_gauss   # [N, K]
        return torch.logsumexp(log_components, dim=-1)                  # [N]

    def _sample(self, num_samples: int, context=None) -> torch.Tensor:
        """
        Sample from the GMM:
            1. Pick component k ~ Categorical(π)
            2. Draw z ~ N(μ_k, diag(σ²_k))
        """
        device = self._means.device

        if not self._fitted:
            return torch.randn(num_samples, self.feature_dim, device=device)

        weights = self._log_weights.exp()
        weights = weights / weights.sum()                                       # normalise
        k_idx   = torch.multinomial(weights, num_samples, replacement=True)    # [N]
        mu  = self._means[k_idx]                         # [N, D]
        std = (self._log_vars[k_idx] * 0.5).exp()        # [N, D]
        return mu + std * torch.randn_like(mu)
