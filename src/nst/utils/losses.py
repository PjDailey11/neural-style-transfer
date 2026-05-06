"""Loss functions for Gatys NST, AdaIN training, and temporal video coherence."""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn.functional as F

from nst.utils.gram import gram_matrix


def feat_gen_device_zero(gen_feats: dict[str, torch.Tensor]) -> torch.Tensor:
    """Scalar zero on correct device/dtype from feature dict."""
    first = next(iter(gen_feats.values()))
    return torch.zeros((), device=first.device, dtype=first.dtype)


def content_loss(feat_gen: torch.Tensor, feat_content: torch.Tensor) -> torch.Tensor:
    """Mean squared error between content and generated activations."""
    return F.mse_loss(feat_gen, feat_content)


def style_loss_from_features(
    gen_feats: dict[str, torch.Tensor],
    style_feats: dict[str, torch.Tensor],
    layer_weights: dict[str, float] | None = None,
) -> torch.Tensor:
    """Style loss as weighted sum of MSE between Gram matrices."""
    if layer_weights is None:
        layer_weights = {k: 1.0 for k in gen_feats}
    total = feat_gen_device_zero(gen_feats)
    for name in gen_feats:
        if name not in style_feats:
            continue
        g_gen = gram_matrix(gen_feats[name])
        g_style = gram_matrix(style_feats[name]).detach()
        w = layer_weights.get(name, 1.0)
        total = total + w * F.mse_loss(g_gen, g_style)
    return total


def perceptual_content_loss(
    feats_gen: dict[str, torch.Tensor],
    feats_tgt: dict[str, torch.Tensor],
    layers: Iterable[str],
) -> torch.Tensor:
    """Sum of MSEs over selected layers (Huang-style perceptual content term)."""
    total = feat_gen_device_zero(feats_gen)
    for name in layers:
        total = total + F.mse_loss(feats_gen[name], feats_tgt[name])
    return total


def adain_training_losses(
    decoder_out: torch.Tensor,
    content_img: torch.Tensor,
    style_img: torch.Tensor,
    encoder: torch.nn.Module,
    content_layers: tuple[str, ...],
    style_layers: tuple[str, ...],
    lambda_style: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Simplified AdaIN training objective (decoder-only training).

    Args:
        decoder_out: Output of decoder after AdaIN, same spatial size as ``content_img``.
        content_img: Original content image (normalized).
        style_img: Style reference image (normalized).
        encoder: Frozen encoder producing multi-layer activations.
        content_layers: Layers used for content reconstruction term.
        style_layers: Layers used for Gram matching on decoder output vs style reference.
        lambda_style: Weight on style Gram loss.

    Returns:
        Scalar loss and diagnostics dict.
    """
    gen_feats = encoder(decoder_out)
    c_feats = encoder(content_img)
    s_feats = encoder(style_img)

    c_loss = perceptual_content_loss(gen_feats, c_feats, content_layers)

    gen_style_pick = {k: gen_feats[k] for k in style_layers if k in gen_feats}
    style_pick = {k: s_feats[k] for k in style_layers if k in s_feats}
    s_loss = style_loss_from_features(gen_style_pick, style_pick)

    loss = c_loss + lambda_style * s_loss
    diag = {
        "loss_total": float(loss.detach()),
        "loss_content": float(c_loss.detach()),
        "loss_style": float(s_loss.detach()),
    }
    return loss, diag


def reconet_temporal_loss(
    stylized_t: torch.Tensor,
    stylized_t_prev_warped: torch.Tensor,
    occlusion_weight: torch.Tensor,
    alpha_charbonnier: float = 1e-6,
) -> torch.Tensor:
    """Occlusion-weighted Charbonnier loss between current frame and warped previous.

    Args:
        stylized_t: Current stylized frame ``(N,3,H,W)``.
        stylized_t_prev_warped: Previous stylized frame warped to current coords (same shape).
        occlusion_weight: Weight map ``(N,1,H,W)`` in ``[0,1]`` (higher = trust warp).
        alpha_charbonnier: Numerical stability constant inside square root.

    Returns:
        Scalar temporal coherence loss.
    """
    diff = stylized_t - stylized_t_prev_warped
    sqrt_sum = torch.sqrt(diff * diff + alpha_charbonnier)
    weighted = sqrt_sum.mean(dim=1, keepdim=True) * occlusion_weight
    return weighted.sum() / (occlusion_weight.sum() + 1e-8)
