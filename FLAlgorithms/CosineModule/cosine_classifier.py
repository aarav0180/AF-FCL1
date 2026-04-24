"""
Cosine-head variants of the two classifier architectures, with optional
feature calibration (BatchNorm before cosine head) and angular margin.

S_ConvNetCosine  — S_ConvNet  with fc_classifier replaced by CosineLinear
ResnetPlusCosine — Resnet_plus with fc_classifier replaced by CosineLinear

Feature Calibration
-------------------
When feature_calibration=True, a BatchNorm1d layer is inserted between the
fc2 feature layer and the cosine head. This normalizes the feature distribution
across batches, which helps because:
  1. Flow-generated replay features may have different statistics than real data
  2. Different tasks produce features with different mean/variance
  3. BatchNorm acts as a light regularizer that smooths the loss landscape

The BN layer is applied inside forward_from_xa() so it is transparent to the
flow (which operates on xa, before fc2).
"""

from torch import nn
from torch.nn import functional as F

from FLAlgorithms.PreciseFCLNet.classify_net import S_ConvNet, Resnet_plus
from FLAlgorithms.CosineModule.cosine_head import CosineLinear


class S_ConvNetCosine(S_ConvNet):
    """S_ConvNet with a CosineLinear classification head and optional feature calibration."""

    def __init__(self, image_size, image_channel_size, channel_size,
                 xa_dim, num_classes=10, sigma_init=10.0, margin=0.0,
                 feature_calibration=False):
        # Build the full parent network (including the standard fc_classifier)
        super().__init__(image_size, image_channel_size, channel_size,
                         xa_dim=xa_dim, num_classes=num_classes)
        # Replace the final linear layer with the cosine head — same shape
        self.fc_classifier = CosineLinear(xa_dim, num_classes,
                                          sigma_init=sigma_init, margin=margin)
        # Optional feature calibration
        self.feature_calibration = feature_calibration
        if feature_calibration:
            self.feat_bn = nn.BatchNorm1d(xa_dim, affine=True)

    def forward(self, x, labels=None):
        xa = self.forward_to_xa(x)
        classes_p, logits = self.forward_from_xa(xa, labels=labels)
        return classes_p, xa, logits

    def forward_from_xa(self, xa, labels=None):
        xb = F.leaky_relu(self.fc2(xa))
        if self.feature_calibration and hasattr(self, 'feat_bn'):
            xb = self.feat_bn(xb)
        logits = self.fc_classifier(xb, labels=labels)
        classes_p = self.softmax(logits)
        return classes_p, logits


class ResnetPlusCosine(Resnet_plus):
    """Resnet_plus with a CosineLinear classification head and optional feature calibration."""

    def __init__(self, image_size, xa_dim, num_classes=10, sigma_init=10.0,
                 margin=0.0, feature_calibration=False):
        # Build the full parent network (including the standard fc_classifier)
        super().__init__(image_size, xa_dim=xa_dim, num_classes=num_classes)
        # Replace the final linear layer with the cosine head — same shape
        self.fc_classifier = CosineLinear(xa_dim, num_classes,
                                          sigma_init=sigma_init, margin=margin)
        # Optional feature calibration
        self.feature_calibration = feature_calibration
        if feature_calibration:
            self.feat_bn = nn.BatchNorm1d(xa_dim, affine=True)

    def forward(self, x, labels=None):
        xa = self.forward_to_xa(x)
        classes_p, logits = self.forward_from_xa(xa, labels=labels)
        return classes_p, xa, logits

    def forward_from_xa(self, xa, labels=None):
        xa = F.leaky_relu(self.fc1(xa))
        xb = F.leaky_relu(self.fc2(xa))
        if self.feature_calibration and hasattr(self, 'feat_bn'):
            xb = self.feat_bn(xb)
        logits = self.fc_classifier(xb, labels=labels)
        classes_p = self.softmax(logits)
        return classes_p, logits
