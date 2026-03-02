"""
KLReg model variants — composable with PreciseModel and GMMPreciseModel.

MRO for KLRegPreciseModel:
    KLRegPreciseModel → KLRegMixin → PreciseModel → nn.Module

MRO for KLRegGMMPreciseModel:
    KLRegGMMPreciseModel → KLRegMixin → GMMPreciseModel → PreciseModel → nn.Module

In both cases KLRegMixin.train_a_batch_flow shadows PreciseModel's version;
every other method (train_a_batch_classifier, fit_gmm_prior, …) falls through
to the appropriate base class unchanged.
"""

from FLAlgorithms.KLRegModule.klreg_mixin import KLRegMixin
from FLAlgorithms.PreciseFCLNet.model import PreciseModel
from FLAlgorithms.GMMModule.gmm_model import GMMPreciseModel


class KLRegPreciseModel(KLRegMixin, PreciseModel):
    """PreciseFCL + stabilised flow training (gradient clip + optional Jacobian KL)."""

    def __init__(self, args):
        super().__init__(args)          # → KLRegMixin (no __init__) → PreciseModel.__init__
        self.klreg_beta = args.klreg_beta
        self.klreg_clip = args.klreg_clip


class KLRegGMMPreciseModel(KLRegMixin, GMMPreciseModel):
    """GMM prior + stabilised flow training (gradient clip + optional Jacobian KL)."""

    def __init__(self, args):
        super().__init__(args)          # → KLRegMixin (no __init__) → GMMPreciseModel.__init__
        self.klreg_beta = args.klreg_beta
        self.klreg_clip = args.klreg_clip
