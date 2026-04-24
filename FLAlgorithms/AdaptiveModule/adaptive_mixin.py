"""
AdaptiveMixin — accuracy-based adaptive KD weighting.

Problem solved
--------------
During continual learning, the knowledge-distillation weight is a fixed
hyperparameter.  When the model is struggling on the current task (low batch
accuracy) the fixed KD term acts as a rigid constraint that competes with
learning.  When accuracy is already high the same weight may under-regularise.

Solution
--------
Scale the KD loss by

    alpha = sigmoid(batch_acc - 0.5)

where ``batch_acc`` is the fraction of correctly-classified samples in the
current mini-batch (computed from the actual ``softmax_output`` inside
``train_a_batch_classifier``, so zero extra forward passes).

    batch_acc → 0.0  ⟹  alpha ≈ 0.38   (relax KD, let model learn)
    batch_acc → 0.5  ⟹  alpha = 0.50   (balanced)
    batch_acc → 1.0  ⟹  alpha ≈ 0.62   (high accuracy, tighten KD)

When ``last_classifier is None`` (task 0) alpha is forced to 1.0 —
semantically equivalent to the baseline because all KD losses are 0 anyway.

Implementation detail
---------------------
``knowledge_distillation_on_xa_output`` is the single call-site for all four
KD loss components inside ``train_a_batch_classifier``.  By overriding that
one method we intercept KD scaling without duplicating any training logic.

``softmax_output`` is already available as an argument to
``knowledge_distillation_on_xa_output``, so we derive ``batch_acc`` from it
(using ``self._adaptive_y`` stored at the start of the batch).

Usage (via model composition — see adaptive_models.py):

    class AdaptivePreciseModel(AdaptiveMixin, PreciseModel): ...
    class AdaptiveKLRegGMMPreciseModel(AdaptiveMixin, KLRegGMMPreciseModel): ...
"""

import torch


class AdaptiveMixin:
    """
    Mixin that overrides ``train_a_batch_classifier`` (to stash ``y``) and
    ``knowledge_distillation_on_xa_output`` (to apply alpha scaling).

    No extra constructor args; ``_adaptive_y`` / ``_adaptive_alpha`` are
    initialised to safe defaults in ``__init__`` for robustness.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._adaptive_y = None
        self._adaptive_alpha = 1.0

    def train_a_batch_classifier(
        self, x, y, flow, last_classifier, global_classifier,
        classes_past_task, available_labels, prototype_bank=None
    ):
        # Stash y and reset alpha so knowledge_distillation_on_xa_output can
        # compute it lazily from the actual softmax_output (no extra fwd pass).
        self._adaptive_y = y
        self._adaptive_alpha = None
        return super().train_a_batch_classifier(
            x, y, flow, last_classifier, global_classifier,
            classes_past_task, available_labels, prototype_bank=prototype_bank,
        )

    def knowledge_distillation_on_xa_output(
        self, x, xa, softmax_output, last_classifier, global_classifier
    ):
        # Compute alpha on first call for this batch (lazy, no extra fwd pass)
        if self._adaptive_alpha is None:
            if last_classifier is not None:
                with torch.no_grad():
                    preds = torch.argmax(softmax_output.detach(), dim=1)
                    batch_acc = (preds == self._adaptive_y).float().mean()
                self._adaptive_alpha = torch.sigmoid(
                    batch_acc - 0.5
                ).item()
            else:
                # Task 0: no last model → KD losses are 0 anyway; alpha is moot
                self._adaptive_alpha = 1.0

        alpha = self._adaptive_alpha

        feat_last, out_last, feat_global, out_global = (
            super().knowledge_distillation_on_xa_output(
                x, xa, softmax_output, last_classifier, global_classifier
            )
        )

        # Scale all four components uniformly.
        # Works for both integer 0 (when a term is disabled) and tensors.
        return feat_last * alpha, out_last * alpha, feat_global * alpha, out_global * alpha
