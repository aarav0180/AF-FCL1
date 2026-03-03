"""
CosineLinear — drop-in replacement for nn.Linear that uses cosine similarity
instead of a raw dot product for the final classification head.

Why this helps continual learning
-----------------------------------
A standard nn.Linear computes  y = x W^T + b.
During training on a new task the optimiser inflates the weight magnitudes
for the new classes to minimise the cross-entropy loss. Old-class weights
stay frozen (or change only via KD), so their magnitudes fall behind.
At test time new classes win simply because ||W_new|| >> ||W_old||.

CosineLinear normalises both the feature vector x and each prototype W_k
onto the unit hypersphere before computing the similarity:

    y_k = σ · (x / ||x||) · (W_k / ||W_k||)

Because magnitudes are divided out, no class can dominate by being "bigger".
The model must learn the *direction* (geometry) of each class in feature
space, which is a strictly harder and more stable objective.

Temperature parameter σ (self.sigma)
--------------------------------------
Raw cosine values lie in [-1, +1].  If we fed them directly into NLLLoss
the softmax distribution would be almost uniform (blurry), stalling learning.
σ (learnable scalar, initialised to 10.0) stretches the range to [-σ, +σ],
allowing the softmax to become sharp and gradients to flow normally.

σ is a single scalar parameter — it is included in model.named_parameters()
and therefore federated automatically with zero special casing.

Usage
-----
Replace:
    self.fc_classifier = nn.Linear(xa_dim, num_classes)
With:
    self.fc_classifier = CosineLinear(xa_dim, num_classes)

The rest of the forward pass (softmax, NLLLoss) is completely unchanged.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineLinear(nn.Module):
    """
    Cosine-similarity classification head with a learnable temperature.

    Parameters
    ----------
    in_features : int
        Dimension of the input feature vector (xa_dim, typically 512).
    out_features : int
        Number of classes.
    sigma_init : float
        Initial value of the temperature scalar σ.  Default: 10.0.
    """

    def __init__(self, in_features: int, out_features: int, sigma_init: float = 10.0):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        # Class prototypes — same shape as nn.Linear weight matrix
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))

        # Learnable temperature scalar (no bias — cosine head has none)
        self.sigma = nn.Parameter(torch.tensor(sigma_init))

        self._reset_parameters()

    def _reset_parameters(self):
        # Kaiming uniform initialisation (same default as nn.Linear)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape [N, in_features]

        Returns
        -------
        torch.Tensor, shape [N, out_features]
            Scaled cosine similarities (logits).
        """
        # Normalise input features and prototype weights onto unit sphere
        x_norm = F.normalize(x, p=2, dim=1)          # [N, D]
        w_norm = F.normalize(self.weight, p=2, dim=1) # [C, D]

        # Cosine similarity: x_norm @ w_norm^T → [N, C]
        cos_sim = x_norm @ w_norm.t()

        # Scale by temperature
        return self.sigma * cos_sim

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, "
                f"out_features={self.out_features}, "
                f"sigma_init={self.sigma.item():.1f}")
