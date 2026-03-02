"""
KLRegMixin — stabilises normalizing-flow training with two fixes that are
confirmed missing from the baseline train_a_batch_flow:

  1. NaN/Inf guard on ``loss_last_flow``
     (baseline only guards ``loss_data``, so a bad replay sample can
     permanently poison the flow from that round onward)

  2. Gradient clipping on flow parameters (``clip_grad_norm_``)
     (baseline clips *classifier* gradients but has **zero** clipping on
     the flow; confirmed root cause of the loss-explosion seen in run.log
     from R6 task-0 onward)

  3. Optional Jacobian KL regularisation (activated when ``klreg_beta > 0``)
     Uses the Hutchinson estimator to approximate ||J||_F^2 with a single
     random probe vector v ~ N(0,I):

         E_v[ ||J^T v||^2 ] = trace(J J^T) = ||J||_F^2

     The regulariser   ||J||_F^2 - log|det J|   pushes the Jacobian of
     the coupling transforms toward orthogonal-like behaviour, preventing
     individual entries from blowing up while the determinant stays in range.

Usage (via model composition — see klreg_models.py):

    class KLRegPreciseModel(KLRegMixin, PreciseModel): ...

MRO guarantees that KLRegMixin.train_a_batch_flow shadows
PreciseModel.train_a_batch_flow while every other method falls through to
the base class unchanged.
"""

import torch
import torch.nn.functional as F

from utils.utils import myitem


class KLRegMixin:
    """
    Mixin that overrides only ``train_a_batch_flow``.

    Requires the concrete class to have:
      • ``self.klreg_clip``  (float) — max-norm for clip_grad_norm_
      • ``self.klreg_beta``  (float) — weight of Jacobian KL term (0 = off)
    Both are set in ``klreg_models.py`` after the parent ``__init__`` runs.
    """

    def train_a_batch_flow(self, x, y, last_flow, classes_so_far, available_labels_past):
        xa = self.classifier.forward_to_xa(x)
        xa = xa.reshape(xa.shape[0], -1)
        y_one_hot = F.one_hot(y, num_classes=self.num_classes).float()

        # ── NLL on current-task real data ────────────────────────────────────
        loss_data = -self.flow.log_prob(inputs=xa, context=y_one_hot).mean()
        if torch.isnan(loss_data) or torch.isinf(loss_data):
            loss_data = torch.tensor(0.0, device=self.device, requires_grad=True)

        # ── NLL on replay samples from last flow ─────────────────────────────
        if self.algorithm == 'PreciseFCL' and last_flow is not None:
            batch_size = x.shape[0]
            with torch.no_grad():
                flow_xa, label, label_one_hot = self.sample_from_flow(
                    last_flow, available_labels_past, batch_size)
            loss_last_flow = -self.flow.log_prob(
                inputs=flow_xa, context=label_one_hot).mean()
            # Guard — missing in baseline; a bad sample here can kill the flow
            if torch.isnan(loss_last_flow) or torch.isinf(loss_last_flow):
                loss_last_flow = torch.tensor(0.0, device=self.device, requires_grad=True)
        else:
            loss_last_flow = 0
        loss_last_flow = self.k_flow_lastflow * loss_last_flow

        # ── Hutchinson-estimator Jacobian KL regularisation ──────────────────
        jac_kl_val = 0.0
        if self.klreg_beta > 0:
            # Detach xa from the classifier graph so gradients here only
            # affect flow parameters, not the classifier.
            xa_jac = xa.detach().requires_grad_(True)
            z_jac, log_det_jac = self.flow._transform(xa_jac, context=y_one_hot)

            # Single Hutchinson probe: E[||J^T v||^2] = ||J||_F^2
            v = torch.randn_like(xa_jac)
            Jv = torch.autograd.grad(
                (z_jac * v).sum(), xa_jac, create_graph=True
            )[0]

            jac_kl = (Jv ** 2).sum(dim=1).mean() - log_det_jac.mean()
            if torch.isnan(jac_kl) or torch.isinf(jac_kl):
                jac_kl = torch.tensor(0.0, device=self.device)
            else:
                jac_kl_val = jac_kl.item()

            loss = loss_data + loss_last_flow + self.klreg_beta * jac_kl
        else:
            loss = loss_data + loss_last_flow

        self.flow_optimizer.zero_grad()
        loss.backward()
        # *** THE FIX: clip flow gradients — confirmed absent in baseline ***
        torch.nn.utils.clip_grad_norm_(self.flow.parameters(), max_norm=self.klreg_clip)
        self.flow_optimizer.step()

        return {
            'flow_loss': loss_data.item(),
            'flow_loss_last': myitem(loss_last_flow),
            'jac_kl': jac_kl_val,
        }
