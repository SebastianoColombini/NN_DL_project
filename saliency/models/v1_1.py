"""v1.1: VGG-like CNN encoder trained from scratch + 3-branch deep-supervision decoder.

The module classes below are copied verbatim from ``v1.1.ipynb`` so that the
checkpoint state_dict keys match exactly.
"""
import torch
import torch.nn as nn

from .base import SaliencyNet

INPUT_SIZE = (240, 320)  # (H, W) used for training


# 5-layer ENCODER
class Encoder(nn.Module):
  def __init__(self):
    super().__init__()
    self.kernel_size = 3
    self.stride = 1
    self.padding = 1

    self.encoder_cnn1 = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(True),
        nn.Conv2d(64, 64, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(True),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    )

    self.encoder_cnn2 = nn.Sequential(
        nn.Conv2d(64, 128, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(True),
        nn.Conv2d(128, 128, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(True),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    )

    self.encoder_cnn3 = nn.Sequential(
        nn.Conv2d(128, 256, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU(True),
        nn.Conv2d(256, 256, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU(True),
        nn.Conv2d(256, 256, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU(True),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    )

    self.encoder_cnn4 = nn.Sequential(
        nn.Conv2d(256, 512, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=False),
        nn.BatchNorm2d(512),
        nn.ReLU(True),
        nn.Conv2d(512, 512, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=False),
        nn.BatchNorm2d(512),
        nn.ReLU(True),
        nn.Conv2d(512, 512, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=False),
        nn.BatchNorm2d(512),
        nn.ReLU(True),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    )

    self.encoder_cnn5 = nn.Sequential(
        nn.Conv2d(512, 512, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=False),
        nn.BatchNorm2d(512),
        nn.ReLU(True),
        nn.Conv2d(512, 512, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=False),
        nn.BatchNorm2d(512),
        nn.ReLU(True),
        nn.Conv2d(512, 512, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=False),
        nn.BatchNorm2d(512),
        nn.ReLU(True)
    )

  # return 3 latent vectors from layers 3-4-5
  def forward(self, x: torch.Tensor):
    x = self.encoder_cnn1(x)
    x = self.encoder_cnn2(x)
    x = self.encoder_cnn3(x)
    a = x.clone()
    x = self.encoder_cnn4(x)
    b = x.clone()
    x = self.encoder_cnn5(x)
    c = x.clone()

    return [a, b, c]


# DECODER layer-3
class Decoder3(nn.Module):
  def __init__(self):
    super().__init__()
    self.decoder_cnn = nn.Sequential(
        nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0),
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


# DECODER layer-4
class Decoder4(nn.Module):
  def __init__(self):
    super().__init__()
    self.decoder_cnn = nn.Sequential(
        nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0),
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


# DECODER layer-5
class Decoder5(nn.Module):
  def __init__(self):
    super().__init__()
    self.decoder_cnn = nn.Sequential(
        nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0),
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


def build_v1_1() -> SaliencyNet:
    return SaliencyNet(Encoder(), Decoder3(), Decoder4(), Decoder5(), FinalCov())
