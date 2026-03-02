"""
Adaptive KD model variants — all 4 flag combinations that include --adaptive.

MRO table
---------
AdaptivePreciseModel:
    AdaptiveMixin → PreciseModel → nn.Module

AdaptiveGMMPreciseModel:
    AdaptiveMixin → GMMPreciseModel → PreciseModel → nn.Module

AdaptiveKLRegPreciseModel:
    AdaptiveMixin → KLRegPreciseModel → KLRegMixin → PreciseModel → nn.Module

AdaptiveKLRegGMMPreciseModel:
    AdaptiveMixin → KLRegGMMPreciseModel → KLRegMixin
               → GMMPreciseModel → PreciseModel → nn.Module

In every case:
  • train_a_batch_flow   → KLRegMixin (if klreg) else PreciseModel
  • train_a_batch_classifier  → AdaptiveMixin → (chain) → PreciseModel
  • knowledge_distillation_on_xa_output → AdaptiveMixin → PreciseModel
  • fit_gmm_prior        → GMMPreciseModel (if gmm)
"""

from FLAlgorithms.AdaptiveModule.adaptive_mixin import AdaptiveMixin
from FLAlgorithms.PreciseFCLNet.model import PreciseModel
from FLAlgorithms.GMMModule.gmm_model import GMMPreciseModel
from FLAlgorithms.KLRegModule.klreg_models import KLRegPreciseModel, KLRegGMMPreciseModel


class AdaptivePreciseModel(AdaptiveMixin, PreciseModel):
    """Baseline + adaptive KD weighting."""
    pass


class AdaptiveGMMPreciseModel(AdaptiveMixin, GMMPreciseModel):
    """GMM prior + adaptive KD weighting."""
    pass


class AdaptiveKLRegPreciseModel(AdaptiveMixin, KLRegPreciseModel):
    """Stabilised flow (klreg) + adaptive KD weighting."""
    pass


class AdaptiveKLRegGMMPreciseModel(AdaptiveMixin, KLRegGMMPreciseModel):
    """GMM prior + stabilised flow (klreg) + adaptive KD weighting."""
    pass
