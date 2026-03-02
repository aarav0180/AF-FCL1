"""
GMMPreciseModel — PreciseModel with TaskGMMPrior as the flow's base distribution.

The only differences from PreciseModel:

  1. After construction, self.flow._distribution is replaced with a
     TaskGMMPrior instance.  The AffineCouplingTransform stack is untouched;
     only the base p(z) changes.

  2. fit_gmm_prior(trainloader, device) runs the full training set through
     the classifier encoder and the flow's transform to collect latent codes
     z = f(xa | y), then calls gmm_prior.fit(z) to update the K-component
     Gaussian mixture.

Everything else — train_a_batch, sample_from_flow, replay CE loss, last_flow
distillation, KD losses — is inherited unchanged from PreciseModel.

The swap is transparent to all callers because TaskGMMPrior implements the
same nflows Distribution interface (log_prob / sample) as StandardNormal.

How it anchors semantic continuity
-----------------------------------
Task T ends  → fit_gmm_prior captures the K-cluster structure of task T's
               latent space inside the GMM prior.
             → UserPreciseFCL.next_task() deepcopies the model (with fitted
               GMM) into last_copy.
Task T+1 trains flow →
   loss = -flow.log_prob(xa_new | y_new).mean()
        = -log p_GMM(z_new) - log|det J_f|
   The new task's features are pushed to be distributed like the previous
   task's multi-modal clusters, preventing semantic drift.
"""

import numpy as np
import torch
import torch.nn.functional as F
import glog as logger

from FLAlgorithms.PreciseFCLNet.model import PreciseModel
from FLAlgorithms.GMMModule.gmm_prior import TaskGMMPrior


class GMMPreciseModel(PreciseModel):

    def __init__(self, args):
        super().__init__(args)
        self.gmm_k = args.gmm_k

        if self.flow is not None:
            feature_dim = int(np.prod(self.xa_shape))
            # Build the GMM prior and replace StandardNormal inside the flow.
            # TaskGMMPrior starts unfitted (falls back to N(0,I) for task 0).
            self._gmm_prior = TaskGMMPrior(feature_dim=feature_dim, K=self.gmm_k)
            self.flow._distribution = self._gmm_prior
            logger.info(
                "GMM prior installed in flow: K=%d components, feature_dim=%d",
                self.gmm_k, feature_dim,
            )
        else:
            self._gmm_prior = None

    # ------------------------------------------------------------------
    # Device movement — GMM buffers travel with the flow automatically
    # because TaskGMMPrior is registered as flow._distribution (an nn.Module
    # submodule).  We call it explicitly here for clarity.
    # ------------------------------------------------------------------

    def to(self, device):
        super().to(device)        # moves classifier + flow (and thus _gmm_prior)
        return self

    # ------------------------------------------------------------------
    # GMM fitting — called by GMMUserPreciseFCL.next_task() BEFORE deepcopy
    # ------------------------------------------------------------------

    def fit_gmm_prior(self, trainloader, device):
        """
        Collect latent codes z = flow._transform(xa | y) for every sample in
        trainloader, then refit the GMM prior on those codes.

        Must be called while the model still holds task-T weights (before the
        deepcopy that creates last_copy), so the transform is consistent with
        the features the current task produced.

        Parameters
        ----------
        trainloader : DataLoader
            Yields (x, y) tuples.  Typically self.trainloaderfull.
        device : torch.device
        """
        if self.flow is None or self._gmm_prior is None:
            return

        self.classifier.eval()
        self.flow.eval()

        z_list = []
        with torch.no_grad():
            for x_batch, y_batch in trainloader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                xa = self.classifier.forward_to_xa(x_batch)
                xa = xa.reshape(xa.shape[0], -1)                      # [N, 512]

                # Class-conditioned context (same as during flow training)
                y_one_hot = F.one_hot(y_batch, num_classes=self.num_classes).float()
                embedded_ctx = self.flow._embedding_net(y_one_hot)

                # Latent code under the CURRENT transform (no log-det needed)
                z, _ = self.flow._transform(xa, context=embedded_ctx)  # [N, 512]
                z_list.append(z.cpu().numpy())

        z_all = np.concatenate(z_list, axis=0)   # [N_total, 512]
        logger.info(
            "Fitting GMM prior on %d latent codes (K=%d)...",
            z_all.shape[0], self.gmm_k,
        )
        self._gmm_prior.fit(z_all)
        logger.info("GMM prior fitted successfully.")
