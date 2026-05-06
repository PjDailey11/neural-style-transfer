"""Fast neural style transfer with Adaptive Instance Normalization (Huang et al.)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

from nst.utils.image_tensor import rgb01_to_imagenet

# Encoder indices up to ``relu4_1`` (inclusive) in torchvision VGG-19 ``features``.
_ENCODER_DEPTH = 21


def adaptive_instance_normalization(content_feat: torch.Tensor, style_feat: torch.Tensor) -> torch.Tensor:
    """Apply AdaIN: transfer mean/variance of ``style_feat`` onto ``content_feat``.

    Shapes: ``(N,C,H_c,W_c)`` and ``(N,C,H_s,W_s)``.
    """
    assert content_feat.size()[:2] == style_feat.size()[:2], "Channel mismatch for AdaIN."
    style_mean = style_feat.mean(dim=(2, 3), keepdim=True)
    style_std = style_feat.std(dim=(2, 3), keepdim=True) + 1e-5
    content_mean = content_feat.mean(dim=(2, 3), keepdim=True)
    content_std = content_feat.std(dim=(2, 3), keepdim=True) + 1e-5
    normalized = (content_feat - content_mean) / content_std
    return normalized * style_std + style_mean


class AdaINEncoder(nn.Module):
    """First layers of VGG-19 up to ``relu4_1``, emitting intermediate activations."""

    RELU_INDICES: tuple[int, ...] = (1, 3, 6, 8, 11, 13, 15, 17, 20)
    NAMES: tuple[str, ...] = (
        "relu1_1",
        "relu1_2",
        "relu2_1",
        "relu2_2",
        "relu3_1",
        "relu3_2",
        "relu3_3",
        "relu3_4",
        "relu4_1",
    )

    def __init__(self, weights: models.VGG19_Weights | None = models.VGG19_Weights.IMAGENET1K_V1) -> None:
        super().__init__()
        self.net = models.vgg19(weights=weights).features[:_ENCODER_DEPTH].eval()
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        acts: dict[str, torch.Tensor] = {}
        h = x
        relu_cursor = 0
        for idx, layer in enumerate(self.net):
            h = layer(h)
            if idx in self.RELU_INDICES:
                acts[self.NAMES[relu_cursor]] = h
                relu_cursor += 1
        return acts


class Decoder(nn.Module):
    """Mirror decoder from ``relu4_1`` (512 channels) back to RGB."""

    def __init__(self) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = 512

        def block(in_c: int, out_c: int, upsample: bool) -> None:
            nonlocal layers
            layers.append(nn.ReflectionPad2d(1))
            layers.append(nn.Conv2d(in_c, out_c, kernel_size=3))
            layers.append(nn.ReLU(inplace=True))
            if upsample:
                layers.append(nn.Upsample(scale_factor=2, mode="nearest"))

        block(in_ch, 256, True)
        block(256, 256, False)
        block(256, 256, False)
        block(256, 256, False)
        block(256, 128, True)
        block(128, 128, False)
        block(128, 64, True)
        block(64, 64, False)
        layers.extend(
            [
                nn.ReflectionPad2d(1),
                nn.Conv2d(64, 3, kernel_size=3),
            ]
        )
        self.decoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


class AdaINNet(nn.Module):
    """Encoder + AdaIN + Decoder for arbitrary style transfer."""

    def __init__(
        self,
        encoder_weights: models.VGG19_Weights | None = models.VGG19_Weights.IMAGENET1K_V1,
    ) -> None:
        super().__init__()
        self.encoder = AdaINEncoder(weights=encoder_weights)
        self.decoder = Decoder()

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        feats = self.encoder(x)
        return feats["relu4_1"], feats

    @torch.no_grad()
    def stylize(self, content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """Single forward stylization in eval mode."""
        self.eval()
        cf, _ = self.encode(content)
        sf, _ = self.encode(style)
        ad = adaptive_instance_normalization(cf, sf)
        rgb = torch.sigmoid(self.decoder(ad))
        return rgb01_to_imagenet(rgb)

    def forward_train(self, content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """Training forward returning ImageNet-normalized prediction."""
        cf, _ = self.encode(content)
        sf, _ = self.encode(style)
        ad = adaptive_instance_normalization(cf, sf)
        rgb = torch.sigmoid(self.decoder(ad))
        return rgb01_to_imagenet(rgb)
