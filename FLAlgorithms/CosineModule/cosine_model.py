"""
Cosine-head model variants — CosineMixin swaps the classifier head, composable
with all existing mixins (GMM, KLReg, Adaptive).

All angular-geometry fixes are OPTIONAL and OFF by default:
  --vmf_prob      → angular replay-selection (vMF instead of Gaussian)
  --angular_kd    → cosine-distance feature KD (instead of L2)

Without these flags, CosineMixin behaves identically to the original
cosine implementation (classifier swap only, Euclidean replay/KD).
"""

import numpy as np
import torch
import torch.nn.functional as F
import glog as logger

from FLAlgorithms.CosineModule.cosine_classifier import S_ConvNetCosine, ResnetPlusCosine
from FLAlgorithms.PreciseFCLNet.model import PreciseModel, MultiClassCrossEntropy
from FLAlgorithms.GMMModule.gmm_model import GMMPreciseModel
from FLAlgorithms.KLRegModule.klreg_models import KLRegPreciseModel, KLRegGMMPreciseModel
from FLAlgorithms.AdaptiveModule.adaptive_models import (
    AdaptivePreciseModel,
    AdaptiveGMMPreciseModel,
    AdaptiveKLRegPreciseModel,
    AdaptiveKLRegGMMPreciseModel,
)

eps = 1e-30


class CosineMixin:
    """
    Mixin that swaps the classification head to CosineLinear.

    Optional angular-geometry overrides (all default OFF):
      vmf_prob:   override probability_in_localdata → von Mises–Fisher
      angular_kd: override feature KD → cosine distance
    """

    def __init__(self, args):
        super().__init__(args)

        sigma_init = getattr(args, 'cosine_sigma', 10.0)
        xa_dim     = int(np.prod(self.xa_shape))
        dataset    = args.dataset

        # --- Feature flags (all default OFF) ---
        self.vmf_prob   = getattr(args, 'vmf_prob', False)
        self.angular_kd = getattr(args, 'angular_kd', False)
        self.vmf_kappa_min = getattr(args, 'vmf_kappa_min', 0.5)
        self.vmf_kappa_max = getattr(args, 'vmf_kappa_max', 50.0)

        if self.vmf_prob:
            logger.info('[CosineMixin] Angular replay-selection ENABLED (vMF, '
                        f'kappa_range=[{self.vmf_kappa_min}, {self.vmf_kappa_max}])')
        if self.angular_kd:
            logger.info('[CosineMixin] Angular feature KD ENABLED (cosine distance)')

        # Swap the classifier in-place
        if 'EMNIST-Letters' in dataset or dataset == 'MNIST-SVHN-FASHION':
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
        import torch.optim as optim
        beta1        = args.beta1
        beta2        = args.beta2
        weight_decay = args.weight_decay
        lr           = args.lr

        self.classifier_optimizer = optim.Adam(
            self.classifier.parameters(),
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

    # ------------------------------------------------------------------
    # OVERRIDE 1: Angular replay-selection (von Mises–Fisher)
    #   Only active when --vmf_prob is passed.
    #   Otherwise falls through to PreciseModel's Gaussian version.
    # ------------------------------------------------------------------
    def probability_in_localdata(self, xa_u, y, prob_mean, flow_xa, flow_label):
        if not self.vmf_prob:
            return super().probability_in_localdata(
                xa_u, y, prob_mean, flow_xa, flow_label
            )

        flow_xa_label_set = set(flow_label)
        flow_xa_prob = torch.zeros([flow_xa.shape[0]], device=flow_xa.device)

        for flow_yi in flow_xa_label_set:
            if (y == flow_yi).sum() > 0:
                xa_u_yi = xa_u[y == flow_yi]
                flow_xa_yi = flow_xa[flow_label == flow_yi]

                # Normalise both to unit sphere
                xa_u_yi_norm = F.normalize(xa_u_yi, p=2, dim=1)
                flow_xa_yi_norm = F.normalize(flow_xa_yi, p=2, dim=1)

                # Mean direction μ
                mu_unnorm = xa_u_yi_norm.mean(dim=0, keepdim=True)
                R_bar = mu_unnorm.norm(dim=1, keepdim=True).clamp(min=1e-6)
                mu = mu_unnorm / R_bar

                # Estimate concentration κ from mean resultant length
                # κ ≈ R̄·(d - R̄²) / (1 - R̄²)
                d = float(xa_u_yi.shape[1])
                R = R_bar.squeeze()
                kappa = R * (d - R * R) / (1.0 - R * R + 1e-6)
                kappa = kappa.clamp(min=self.vmf_kappa_min,
                                    max=self.vmf_kappa_max)

                # vMF probability: exp(κ·(cos - 1))  → [0, 1], max at cos=1
                cos_sim = (flow_xa_yi_norm * mu).sum(dim=1)
                vmf_prob = torch.exp(kappa * (cos_sim - 1.0))

                flow_xa_prob[flow_label == flow_yi] = vmf_prob
            else:
                flow_xa_prob[flow_label == flow_yi] = prob_mean

        return flow_xa_prob

    # ------------------------------------------------------------------
    # OVERRIDE 2: Angular feature KD (cosine distance)
    #   Only active when --angular_kd is passed.
    #   Otherwise falls through to PreciseModel's L2 version.
    # ------------------------------------------------------------------
    def knowledge_distillation_on_xa_output(self, x, xa, softmax_output,
                                             last_classifier, global_classifier):
        if not self.angular_kd:
            return super().knowledge_distillation_on_xa_output(
                x, xa, softmax_output, last_classifier, global_classifier
            )

        if self.k_kd_last_cls > 0 and type(last_classifier) != type(None):
            softmax_output_last, xa_last, _ = last_classifier(x)
            xa_last = xa_last.detach()
            softmax_output_last = softmax_output_last.detach()

            # Angular feature KD: 1 - cos(xa, xa_last)
            kd_loss_feature_last = self.k_kd_last_cls * (
                1.0 - F.cosine_similarity(xa, xa_last, dim=1).mean()
            )
            kd_loss_output_last = self.k_kd_last_cls * MultiClassCrossEntropy(
                softmax_output, softmax_output_last, T=2
            )
        else:
            kd_loss_feature_last = 0
            kd_loss_output_last = 0

        if self.k_kd_global_cls > 0:
            softmax_output_global, xa_global, _ = global_classifier(x)
            xa_global = xa_global.detach()
            softmax_output_global = softmax_output_global.detach()

            # Angular feature KD: 1 - cos(xa, xa_global)
            kd_loss_feature_global = self.k_kd_global_cls * (
                1.0 - F.cosine_similarity(xa, xa_global, dim=1).mean()
            )
            kd_loss_output_global = self.k_kd_global_cls * MultiClassCrossEntropy(
                softmax_output, softmax_output_global, T=2
            )
        else:
            kd_loss_feature_global = 0
            kd_loss_output_global = 0

        return kd_loss_feature_last, kd_loss_output_last, \
               kd_loss_feature_global, kd_loss_output_global


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
