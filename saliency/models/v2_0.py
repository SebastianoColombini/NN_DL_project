"""v2.0: Swin-Tiny encoder (timm) + 3-branch deep-supervision decoder.

The module classes below are copied from ``notebooks/v2.0.ipynb``; the only change is
``pretrained=False`` in the backbone, since the checkpoint holds the full
fine-tuned backbone.
"""
import timm
import torch
import torch.nn as nn

from .base import SaliencyNet

BACKBONE = "swin_tiny_patch4_window7_224"
INPUT_SIZE = (224, 224)  # (H, W) used for training


# Pretrained transformer ENCODER (replaces the from-scratch 5-layer CNN encoder of v1.1)
class TransformerEncoder(nn.Module):
    """Swin-Tiny backbone (ImageNet-1k pretrained).

    Returns 3 feature maps at strides 8/16/32 (channels 192/384/768 for swin_tiny),
    matching the 3-branch deep-supervision decoder from v1.1.
    """
    def __init__(self, backbone_name="swin_tiny_patch4_window7_224",
                 freeze_backbone=True, unfreeze_last_n_stages=1):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=False,  # weights come from the checkpoint; no download
            features_only=True,
            out_indices=(1, 2, 3),
        )
        self.out_channels = self.backbone.feature_info.channels()  # [192, 384, 768]
        self.out_strides = self.backbone.feature_info.reduction()  # [8, 16, 32]

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            # Unfreeze the last N top-level modules (deepest stages) for fine-tuning
            if unfreeze_last_n_stages > 0:
                children = list(self.backbone.named_children())
                for name, module in children[-unfreeze_last_n_stages:]:
                    for p in module.parameters():
                        p.requires_grad = True

    # return 3 hierarchical feature maps (strides 8/16/32)
    def forward(self, x: torch.Tensor):
        a, b, c = self.backbone(x)
        # This Swin backbone returns NHWC tensors, while the Conv2d decoders expect NCHW --> convert explicitly.
        a = a.permute(0, 3, 1, 2).contiguous()
        b = b.permute(0, 3, 1, 2).contiguous()
        c = c.permute(0, 3, 1, 2).contiguous()
        return [a, b, c]


# DECODER layer-3 (stride-8 branch)
class Decoder3(nn.Module):
  def __init__(self, in_channels=192):
    super().__init__()
    self.decoder_cnn = nn.Sequential(
        nn.Conv2d(in_channels, 64, kernel_size=1, stride=1, padding=0),
        nn.BatchNorm2d(64),
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        nn.Conv2d(64, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(True),
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        nn.Conv2d(32, 16, kernel_size=3, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(True),
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        nn.Conv2d(16, 1, kernel_size=3, padding=1),
        nn.BatchNorm2d(1),
        nn.ReLU(True)
    )

  def forward(self, x: torch.Tensor):
    return self.decoder_cnn(x)


# DECODER layer-4 (stride-16 branch)
class Decoder4(nn.Module):
  def __init__(self, in_channels=384):
    super().__init__()
    self.decoder_cnn = nn.Sequential(
        nn.Conv2d(in_channels, 128, kernel_size=1, stride=1, padding=0),
        nn.BatchNorm2d(128),
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        nn.Conv2d(128, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(True),
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        nn.Conv2d(64, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(True),
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        nn.Conv2d(32, 16, kernel_size=3, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(True),
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        nn.Conv2d(16, 1, kernel_size=3, padding=1),
        nn.BatchNorm2d(1),
        nn.ReLU(True)
    )

  def forward(self, x: torch.Tensor):
    return self.decoder_cnn(x)


# DECODER layer-5 (stride-32 branch)
class Decoder5(nn.Module):
  def __init__(self, in_channels=768):
    super().__init__()
    self.decoder_cnn = nn.Sequential(
        nn.Conv2d(in_channels, 128, kernel_size=1, stride=1, padding=0),
        nn.BatchNorm2d(128),
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        nn.Conv2d(128, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(True),
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        nn.Conv2d(64, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(True),
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        nn.Conv2d(32, 16, kernel_size=3, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(True),
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        nn.Conv2d(16, 16, kernel_size=3, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(True),
        # extra stage vs v1.1: input here is stride-32 (not stride-16), so one more x2 is needed to reach full resolution
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        nn.Conv2d(16, 1, kernel_size=3, padding=1),
        nn.BatchNorm2d(1),
        nn.ReLU(True)
    )

  def forward(self, x: torch.Tensor):
    return self.decoder_cnn(x)


# Final layer to combine the 3 saliency maps
class FinalCov(nn.Module):
  def __init__(self):
    super().__init__()
    self.decoder_cnn = nn.Sequential(
        nn.ConvTranspose2d(3, 1, kernel_size=1, stride=1, padding=0),
        nn.Sigmoid()
    )

  def forward(self, x: torch.Tensor):
    return self.decoder_cnn(x)


def build_v2_0() -> SaliencyNet:
    enc = TransformerEncoder(BACKBONE, freeze_backbone=True, unfreeze_last_n_stages=1)
    c3, c4, c5 = enc.out_channels
    return SaliencyNet(enc, Decoder3(c3), Decoder4(c4), Decoder5(c5), FinalCov())
