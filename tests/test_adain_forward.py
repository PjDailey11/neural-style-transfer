from __future__ import annotations

import torch

from nst.models.adain_net import AdaINEncoder, AdaINNet
from nst.models.reconet_style import ReCoNetStyle


def test_adain_encoder_dict_keys() -> None:
    enc = AdaINEncoder(weights=None).eval()
    x = torch.randn(1, 3, 64, 64)
    feats = enc(x)
    assert "relu4_1" in feats
    assert feats["relu4_1"].shape[1] == 512


def test_adain_net_stylize() -> None:
    torch.manual_seed(0)
    net = AdaINNet(encoder_weights=None).eval()
    c = torch.randn(1, 3, 64, 64)
    s = torch.randn(1, 3, 48, 72)
    out = net.stylize(c, s)
    assert out.shape == c.shape


def test_reconet_forward() -> None:
    torch.manual_seed(0)
    m = ReCoNetStyle().eval()
    x = torch.randn(1, 3, 96, 96)
    y = m(x)
    assert y.shape == (1, 3, 96, 96)
