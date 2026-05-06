"""Gram matrix utilities for style representation."""

from __future__ import annotations

import torch


def gram_matrix(feat: torch.Tensor) -> torch.Tensor:
    """Compute Gram matrix G for feature maps shaped (N, C, H, W).

    G[c,d] = sum_{h,w} F[c,h,w] * F[d,h,w] / (C_h * C_w) — here we use unnormalized
    Frobenius inner product scaled by spatial size for numerical stability matching common NST code.

    Args:
        feat: Activations of shape ``(batch, channels, height, width)``.

    Returns:
        Gram matrices of shape ``(batch, channels, channels)``.
    """
    if feat.dim() != 4:
        raise ValueError(f"Expected 4D tensor (N,C,H,W), got shape {tuple(feat.shape)}")
    n, c, h, w = feat.shape
    f = feat.view(n, c, h * w)
    gram = torch.bmm(f, f.transpose(1, 2))
    return gram / (c * h * w)
