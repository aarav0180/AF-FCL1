"""
Cosine-head variants of the two classifier architectures.

S_ConvNetCosine  — S_ConvNet  with fc_classifier replaced by CosineLinear
ResnetPlusCosine — Resnet_plus with fc_classifier replaced by CosineLinear

Only __init__ is overridden in each case.  Every other method
(forward, forward_to_xa, forward_from_xa) is inherited verbatim.

The swap is transparent to all callers:
  • forward()         → returns (softmax_p, xa, logits)  — same shape, same semantics
  • forward_from_xa() → returns (softmax_p, logits)      — same shape, same semantics
  • logits now represent scaled cosine similarities instead of dot products,
    but they are still raw pre-softmax scores passed into the same NLLLoss.
"""

from FLAlgorithms.PreciseFCLNet.classify_net import S_ConvNet, Resnet_plus
from FLAlgorithms.CosineModule.cosine_head import CosineLinear


class S_ConvNetCosine(S_ConvNet):
    """S_ConvNet with a CosineLinear classification head."""

    def __init__(self, image_size, image_channel_size, channel_size,
                 xa_dim, num_classes=10, sigma_init=10.0):
        # Build the full parent network (including the standard fc_classifier)
        super().__init__(image_size, image_channel_size, channel_size,
                         xa_dim=xa_dim, num_classes=num_classes)
        # Replace the final linear layer with the cosine head — same shape
        self.fc_classifier = CosineLinear(xa_dim, num_classes, sigma_init=sigma_init)


class ResnetPlusCosine(Resnet_plus):
    """Resnet_plus with a CosineLinear classification head."""

    def __init__(self, image_size, xa_dim, num_classes=10, sigma_init=10.0):
        # Build the full parent network (including the standard fc_classifier)
        super().__init__(image_size, xa_dim=xa_dim, num_classes=num_classes)
        # Replace the final linear layer with the cosine head — same shape
        self.fc_classifier = CosineLinear(xa_dim, num_classes, sigma_init=sigma_init)
