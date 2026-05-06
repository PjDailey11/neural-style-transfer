from __future__ import annotations

import torch

from nst.models.gatys_vgg import GatysVGG


def test_gatys_forward_shapes() -> None:
    torch.manual_seed(0)
    model = GatysVGG(weights=None).eval()
    x = torch.randn(1, 3, 32, 32)
    acts, content = model(x)
    assert content.ndim == 4
    assert len(acts) == len(model.style_layer_names)
