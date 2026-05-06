from __future__ import annotations

import torch

from nst.models.adain_net import AdaINNet
from nst.models.temporal_video import TemporalMode, TemporalStylizationPipeline


def test_temporal_adain_ema_runs() -> None:
    torch.manual_seed(0)
    device = torch.device("cpu")
    net = AdaINNet(encoder_weights=None).eval().to(device)
    pipe = TemporalStylizationPipeline(TemporalMode.ADAIN_EMA, device, adain=net, ema_beta=0.9)
    c = torch.randn(1, 3, 32, 32, device=device)
    s = torch.randn(1, 3, 32, 32, device=device)
    o1 = pipe.process_frame(c, s)
    o2 = pipe.process_frame(c * 0.95 + s * 0.05, s)
    assert o1.shape == (1, 3, 32, 32)
    assert o2.shape == (1, 3, 32, 32)
