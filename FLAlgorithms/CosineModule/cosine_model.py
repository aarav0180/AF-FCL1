"""
Cosine-head model variants — CosineMixin swaps the classifier head, composable
with all existing mixins (GMM, KLReg, Adaptive).

Now includes:
  - Angular margin (ArcFace-style) via --cosine_margin
  - Feature calibration (BatchNorm) via --cosine_calibration
  - EWC-style parameter importance regularization via --cosine_ewc
"""

import numpy as np
import torch

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

    New features:
      - cosine_margin:       Additive angular margin (ArcFace-style)
      - cosine_calibration:  BatchNorm before cosine head
      - cosine_ewc:          EWC-style regularization weight
    """

    def __init__(self, args):
        super().__init__(args)   # PreciseModel (or GMM/KLReg variant) builds everything

        sigma_init = getattr(args, 'cosine_sigma', 10.0)
        margin     = getattr(args, 'cosine_margin', 0.0)
        calibration = getattr(args, 'cosine_calibration', False)
        xa_dim     = int(np.prod(self.xa_shape))
        dataset    = args.dataset

        # EWC importance tracking
        self.cosine_ewc_lambda = getattr(args, 'cosine_ewc', 0.0)
        self._ewc_params = {}    # {name: param_snapshot}
        self._ewc_fisher = {}    # {name: fisher_diagonal}

        # Swap the classifier in-place — preserves all other layers
        if 'EMNIST-Letters' in dataset or dataset == 'MNIST-SVHN-FASHION':
            image_size         = 28 if 'EMNIST' in dataset else 32
            image_channel_size = 1  if 'EMNIST' in dataset else 3
            c_channel_size     = args.c_channel_size
            self.classifier = S_ConvNetCosine(
                image_size, image_channel_size, c_channel_size,
                xa_dim=xa_dim, num_classes=self.num_classes,
                sigma_init=sigma_init, margin=margin,
                feature_calibration=calibration,
            )
        elif dataset == 'CIFAR100':
            self.classifier = ResnetPlusCosine(
                32, xa_dim=xa_dim, num_classes=self.num_classes,
                sigma_init=sigma_init, margin=margin,
                feature_calibration=calibration,
            )

        # Rebuild the classifier optimisers to point at the new classifier.
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

    def compute_ewc_fisher(self, dataloader, device, num_batches=50):
        """
        Compute diagonal Fisher Information Matrix after each task ends.
        Called from the user's next_task() before deepcopy.
        """
        if self.cosine_ewc_lambda <= 0:
            return

        self.classifier.eval()
        fisher = {n: torch.zeros_like(p) for n, p in self.classifier.named_parameters()}

        count = 0
        for x, y in dataloader:
            if count >= num_batches:
                break
            x, y = x.to(device), y.to(device)
            self.classifier.zero_grad()
            p, _, logits = self.classifier(x)
            p = torch.clamp(p, min=1e-30)
            loss = torch.nn.functional.nll_loss(torch.log(p), y)
            loss.backward()
            for n, param in self.classifier.named_parameters():
                if param.grad is not None:
                    fisher[n] += param.grad.data.pow(2)
            count += 1

        # Average and store
        for n in fisher:
            fisher[n] /= max(count, 1)
            # Merge with existing Fisher (running average across tasks)
            if n in self._ewc_fisher:
                self._ewc_fisher[n] = 0.5 * self._ewc_fisher[n] + 0.5 * fisher[n]
            else:
                self._ewc_fisher[n] = fisher[n]

        # Snapshot parameters
        self._ewc_params = {
            n: p.data.clone() for n, p in self.classifier.named_parameters()
        }

    def ewc_penalty(self):
        """Compute EWC penalty term to add to classifier loss."""
        if self.cosine_ewc_lambda <= 0 or len(self._ewc_fisher) == 0:
            return 0.0

        penalty = 0.0
        for n, p in self.classifier.named_parameters():
            if n in self._ewc_fisher and n in self._ewc_params:
                penalty += (self._ewc_fisher[n] * (p - self._ewc_params[n]).pow(2)).sum()
        return self.cosine_ewc_lambda * penalty

    def get_extra_classifier_loss(self):
        """Called by PreciseModel.train_a_batch_classifier before backward.
        Returns extra loss term (EWC penalty) to add to c_loss."""
        return self.ewc_penalty()


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
