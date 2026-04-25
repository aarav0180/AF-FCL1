"""
Cosine-head model variants — INFERENCE-TIME cosine normalization.

KEY DESIGN CHANGE (v2):
  Previous approach replaced the entire classifier with CosineLinear,
  which destroyed the training dynamics that AF-FCL was tuned for.

  New approach: Keep the standard nn.Linear for TRAINING (identical
  gradient dynamics to baseline). Apply cosine normalization ONLY
  during evaluation/inference, which removes magnitude bias when
  it matters (test-time prediction on all classes seen so far).

  This is inspired by LUCIR (Hou et al., CVPR 2019) which showed
  that post-hoc weight normalization is sufficient to eliminate
  the new-class bias in class-incremental learning.

  Additionally, during training we add a lightweight weight-norm
  regularizer that gently encourages prototype norms to stay balanced,
  without changing the logit computation.
"""

import torch
import torch.nn.functional as F

from FLAlgorithms.PreciseFCLNet.model import PreciseModel
from FLAlgorithms.GMMModule.gmm_model import GMMPreciseModel
from FLAlgorithms.KLRegModule.klreg_models import KLRegPreciseModel, KLRegGMMPreciseModel
from FLAlgorithms.AdaptiveModule.adaptive_models import (
    AdaptivePreciseModel,
    AdaptiveGMMPreciseModel,
    AdaptiveKLRegPreciseModel,
    AdaptiveKLRegGMMPreciseModel,
)


class CosineMixin:
    """
    Mixin that applies cosine normalization at INFERENCE TIME only.

    During training:
      - Forward pass uses standard nn.Linear (y = xW^T + b) — IDENTICAL to baseline
      - A lightweight weight-norm balance regularizer is added to the loss

    During evaluation (model.eval()):
      - fc_classifier.forward is monkey-patched to normalize both x and W
      - Logits become σ * cos(θ), removing magnitude bias for test predictions

    This preserves all training dynamics (gradients, flow, KD) while fixing
    the magnitude bias that hurts old-class accuracy at test time.
    """

    def __init__(self, args):
        super().__init__(args)   # PreciseModel builds everything normally

        # Config
        self.cosine_sigma = getattr(args, 'cosine_sigma', 10.0)
        self.cosine_wnorm_lambda = getattr(args, 'cosine_ewc', 0.0)
        # Scale down weight-norm regularizer for CIFAR100
        if args.dataset == 'CIFAR100' and self.cosine_wnorm_lambda > 0:
            self.cosine_wnorm_lambda = min(self.cosine_wnorm_lambda, 10.0)

        # Install the eval-time cosine hook
        self._install_cosine_eval_hook()

    def _install_cosine_eval_hook(self):
        """
        Replace fc_classifier.forward with a version that switches between
        standard linear (training) and cosine-normalized (eval) behavior.

        Uses F.linear with live fc.weight/fc.bias references (not a captured
        bound method) so the correct device is always used after .to(cuda).
        """
        fc = self.classifier.fc_classifier
        sigma = self.cosine_sigma
        parent = self  # reference to access training flag

        def cosine_aware_forward(x):
            if parent.classifier.training:
                # Training: standard linear using LIVE weight/bias tensors
                # (NOT a bound method — reads fc.weight directly each call)
                return F.linear(x, fc.weight, fc.bias)
            else:
                # Eval/Test: cosine-normalized logits
                x_norm = F.normalize(x, p=2, dim=1)
                w_norm = F.normalize(fc.weight, p=2, dim=1)
                cos_sim = x_norm @ w_norm.t()
                return sigma * cos_sim

        fc.forward = cosine_aware_forward

    def get_extra_classifier_loss(self):
        """
        Lightweight weight-norm balance regularizer.
        Encourages all class prototype norms to be similar,
        preventing magnitude drift across tasks.
        
        L_wnorm = λ * Var(||W_c||)
        
        This does NOT change the logits — it's a soft regularizer
        on the weight matrix, keeping norms balanced so that the
        eval-time cosine normalization works optimally.
        """
        if self.cosine_wnorm_lambda <= 0:
            return 0.0

        fc = self.classifier.fc_classifier
        # Compute per-class prototype norms
        w_norms = fc.weight.norm(dim=1)  # [num_classes]
        # Variance of norms — penalize imbalance
        norm_var = w_norms.var()
        return self.cosine_wnorm_lambda * norm_var


# ---------------------------------------------------------------------------
# Concrete model classes — one per flag combination
# ---------------------------------------------------------------------------

class CosinePreciseModel(CosineMixin, PreciseModel):
    """Baseline + cosine eval head."""
    pass

class CosineGMMPreciseModel(CosineMixin, GMMPreciseModel):
    """GMM prior + cosine eval head."""
    pass

class CosineKLRegPreciseModel(CosineMixin, KLRegPreciseModel):
    """Stabilised flow (klreg) + cosine eval head."""
    pass

class CosineAdaptivePreciseModel(CosineMixin, AdaptivePreciseModel):
    """Adaptive KD + cosine eval head."""
    pass

class CosineKLRegGMMPreciseModel(CosineMixin, KLRegGMMPreciseModel):
    """GMM prior + stabilised flow + cosine eval head."""
    pass

class CosineAdaptiveGMMPreciseModel(CosineMixin, AdaptiveGMMPreciseModel):
    """GMM prior + adaptive KD + cosine eval head."""
    pass

class CosineAdaptiveKLRegPreciseModel(CosineMixin, AdaptiveKLRegPreciseModel):
    """Stabilised flow + adaptive KD + cosine eval head."""
    pass

class CosineAdaptiveKLRegGMMPreciseModel(CosineMixin, AdaptiveKLRegGMMPreciseModel):
    """GMM prior + stabilised flow + adaptive KD + cosine eval head."""
    pass
