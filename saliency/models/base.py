"""Wrapper that wires an encoder, the three decoder branches and the fusion layer."""
import torch
import torch.nn as nn


class SaliencyNet(nn.Module):
    """encoder -> (decoder3, decoder4, decoder5) -> concat -> final.

    Submodule names match the per-module checkpoints of the notebooks
    (``encoder_best.pth``, ``decoder3_best.pth``, ...).
    """

    def __init__(self, encoder, decoder3, decoder4, decoder5, final):
        super().__init__()
        self.encoder = encoder
        self.decoder3 = decoder3
        self.decoder4 = decoder4
        self.decoder5 = decoder5
        self.final = final

    def forward_all(self, x: torch.Tensor):
        """Return (final, s3, s4, s5), each (N,1,H,W) at input resolution."""
        a, b, c = self.encoder(x)
        s3 = self.decoder3(a)
        s4 = self.decoder4(b)
        s5 = self.decoder5(c)
        out = self.final(torch.cat([s3, s4, s5], dim=1))
        return out, s3, s4, s5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_all(x)[0]


# Order of the sub-networks, also used as checkpoint file prefixes.
SUBMODULES = ("encoder", "decoder3", "decoder4", "decoder5", "final")
