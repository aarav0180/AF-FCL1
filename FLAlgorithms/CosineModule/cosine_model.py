"""
Cosine-head model variants — CosineMixin swaps the classifier head, composable
with all existing mixins (GMM, KLReg, Adaptive).

Architecture
------------
CosineMixin overrides only the three classifier-construction blocks inside
PreciseModel.__init__ (one per dataset).  Everything else — flow, optimisers,
all training methods, KD losses, GMM fitting — is inherited unchanged.

MRO examples
------------
CosinePreciseModel:
    CosineMixin → PreciseModel → nn.Module

CosineAdaptiveKLRegGMMPreciseModel:
    CosineMixin → AdaptiveMixin → KLRegMixin → GMMPreciseModel → PreciseModel → nn.Module

In every case:
  • classifier construction  → CosineMixin (swap to cosine head)
  • train_a_batch_flow       → KLRegMixin if present, else PreciseModel
  • train_a_batch_classifier → AdaptiveMixin → (chain) → PreciseModel
  • knowledge_distillation   → AdaptiveMixin → PreciseModel
  • fit_gmm_prior            → GMMPreciseModel if present

Flag combinations provided (all 8 cosine × {gmm, klreg, adaptive}):
  CosinePreciseModel
  CosineGMMPreciseModel
  CosineKLRegPreciseModel
  CosineAdaptivePreciseModel
  CosineKLRegGMMPreciseModel
  CosineAdaptiveGMMPreciseModel
  CosineAdaptiveKLRegPreciseModel
  CosineAdaptiveKLRegGMMPreciseModel
"""

import numpy as np

from FLAlgorithms.CosineModule.cosine_classifier import S_ConvNetCosine, ResnetPlusCosine
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
    Mixin that replaces the standard nn.Linear classification head with
    CosineLinear immediately after the parent __init__ has built it.

    Requires the concrete class to have:
      • self.classifier  (set by PreciseModel.__init__)
      • self.xa_shape    (set by PreciseModel.__init__)
      • self.num_classes (set by PreciseModel.__init__)
      • args.cosine_sigma (float, default 10.0) — temperature initialisation
    """

    def __init__(self, args):
        super().__init__(args)   # PreciseModel (or GMM/KLReg variant) builds everything

        sigma_init = getattr(args, 'cosine_sigma', 10.0)
        xa_dim     = int(np.prod(self.xa_shape))
        dataset    = args.dataset

        # Swap the classifier in-place — preserves all other layers
        if 'EMNIST-Letters' in dataset or dataset == 'MNIST-SVHN-FASHION':
            # S_ConvNet family
            image_size         = 28 if 'EMNIST' in dataset else 32
            image_channel_size = 1  if 'EMNIST' in dataset else 3
            c_channel_size     = args.c_channel_size
            self.classifier = S_ConvNetCosine(
                image_size, image_channel_size, c_channel_size,
                xa_dim=xa_dim, num_classes=self.num_classes,
                sigma_init=sigma_init,
            )
        elif dataset == 'CIFAR100':
            self.classifier = ResnetPlusCosine(
                32, xa_dim=xa_dim, num_classes=self.num_classes,
                sigma_init=sigma_init,
            )

        # Rebuild the classifier optimisers to point at the new classifier.
        # The flow optimiser is unaffected.
        import torch.optim as optim
        beta1        = args.beta1
        beta2        = args.beta2
        weight_decay = args.weight_decay
        lr           = args.lr

        classifier_params = list(self.classifier.parameters())
        if getattr(self, 'maft_gate', None) is not None:
            classifier_params += list(self.maft_gate.parameters())

        self.classifier_optimizer = optim.Adam(
            classifier_params,
            lr=lr, weight_decay=weight_decay, betas=(beta1, beta2),
        )
        parameters_fb = [
            p for name, p in self.classifier.named_parameters()
            if 'fc2' in name
        ]
        self.classifier_fb_optimizer = optim.Adam(
            parameters_fb, lr=lr, weight_decay=weight_decay,
            betas=(beta1, beta2),
        )


# ---------------------------------------------------------------------------
# Concrete model classes — one per flag combination
# ---------------------------------------------------------------------------

class CosinePreciseModel(CosineMixin, PreciseModel):
    """Baseline + cosine head."""
    pass


class CosineGMMPreciseModel(CosineMixin, GMMPreciseModel):
    """GMM prior + cosine head."""
    pass


class CosineKLRegPreciseModel(CosineMixin, KLRegPreciseModel):
    """Stabilised flow (klreg) + cosine head."""
    pass


class CosineAdaptivePreciseModel(CosineMixin, AdaptivePreciseModel):
    """Adaptive KD + cosine head."""
    pass


class CosineKLRegGMMPreciseModel(CosineMixin, KLRegGMMPreciseModel):
    """GMM prior + stabilised flow + cosine head."""
    pass


class CosineAdaptiveGMMPreciseModel(CosineMixin, AdaptiveGMMPreciseModel):
    """GMM prior + adaptive KD + cosine head."""
    pass


class CosineAdaptiveKLRegPreciseModel(CosineMixin, AdaptiveKLRegPreciseModel):
    """Stabilised flow + adaptive KD + cosine head."""
    pass


class CosineAdaptiveKLRegGMMPreciseModel(CosineMixin, AdaptiveKLRegGMMPreciseModel):
    """GMM prior + stabilised flow + adaptive KD + cosine head."""
    pass
