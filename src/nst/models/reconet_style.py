"""ReCoNet-inspired residual stylization backbone with optional temporal training hooks."""

from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3),
            nn.InstanceNorm2d(channels, affine=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ReCoNetStyle(nn.Module):
    """Compact encoder–decoder with residuals for real-time per-frame stylization.

    Training-time temporal losses (flow-warp + occlusion) live in ``nst.utils.losses`` and
    ``train.py`` when ``--video-arch reconet`` is selected.
    """

    def __init__(self, in_ch: int = 3, base: int = 32, n_residual: int = 5) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(4),
            nn.Conv2d(in_ch, base, kernel_size=9, stride=1),
            nn.InstanceNorm2d(base, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, base * 2, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(base * 2, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(base * 2, base * 4, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(base * 4, affine=True),
            nn.ReLU(inplace=True),
        )
        res = [ResidualBlock(base * 4) for _ in range(n_residual)]
        self.residual = nn.Sequential(*res)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base * 4, base * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(base * 2, affine=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base * 2, base, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(base, affine=True),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(4),
            nn.Conv2d(base, 3, kernel_size=9, stride=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns stylized image roughly in ``[-1,1]`` — map to ImageNet norm externally."""
        z = self.encoder(x)
        z = self.residual(z)
        return self.decoder(z)


def reconet_output_to_imagenet(y_tanh: torch.Tensor) -> torch.Tensor:
    """Map Tanh output ``[-1,1]`` to approximate ImageNet-normalized tensor."""
    rgb01 = (y_tanh + 1.0) * 0.5
    mean = torch.tensor([0.485, 0.456, 0.406], device=y_tanh.device, dtype=y_tanh.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=y_tanh.device, dtype=y_tanh.dtype).view(1, 3, 1, 1)
    return (rgb01 - mean) / std
