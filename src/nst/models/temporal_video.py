"""Temporal pipelines for stable video stylization."""

from __future__ import annotations

from enum import Enum

import cv2
import numpy as np
import torch

from nst.models.adain_net import AdaINNet
from nst.models.reconet_style import ReCoNetStyle, reconet_output_to_imagenet
from nst.utils.flow import (
    blend_with_previous_flow,
    compute_occlusion_mask,
    flow_to_grid,
    optical_flow_farneback,
    warp_tensor_with_flow,
)
from nst.utils.image_tensor import denormalize_imagenet


class TemporalMode(str, Enum):
    ADAIN_EMA = "adain_ema"
    ADAIN_FLOW = "adain_flow"
    RECONET_FLOW = "reconet_flow"


class TemporalStylizationPipeline:
    """Apply frame-wise stylization with optional temporal coherence."""

    def __init__(
        self,
        mode: TemporalMode,
        device: torch.device,
        adain: AdaINNet | None = None,
        reconet: ReCoNetStyle | None = None,
        ema_beta: float = 0.85,
        flow_blend_beta: float = 0.65,
    ) -> None:
        self.mode = mode
        self.device = device
        self.adain = adain
        self.reconet = reconet
        self.ema_beta = ema_beta
        self.flow_blend_beta = flow_blend_beta
        self._prev_rgb: torch.Tensor | None = None
        self._prev_gray: np.ndarray | None = None

    def reset(self) -> None:
        self._prev_rgb = None
        self._prev_gray = None

    def _stylize_core(self, frame_norm: torch.Tensor, style_norm: torch.Tensor) -> torch.Tensor:
        if self.mode in (TemporalMode.ADAIN_EMA, TemporalMode.ADAIN_FLOW):
            if self.adain is None:
                raise ValueError("AdaIN network required for this temporal mode.")
            return self.adain.stylize(frame_norm, style_norm)
        if self.mode == TemporalMode.RECONET_FLOW:
            if self.reconet is None:
                raise ValueError("ReCoNet network required for reconet_flow mode.")
            rgb01 = denormalize_imagenet(frame_norm)
            recon_in = rgb01 * 2.0 - 1.0
            out_tanh = self.reconet(recon_in)
            return reconet_output_to_imagenet(out_tanh)
        raise ValueError(f"Unsupported temporal mode {self.mode}")

    def process_frame(self, frame_norm: torch.Tensor, style_norm: torch.Tensor) -> torch.Tensor:
        """Process one normalized frame ``(1,3,H,W)``."""
        stylized_norm = self._stylize_core(frame_norm, style_norm)

        rgb_curr = denormalize_imagenet(stylized_norm)

        gray = (
            cv2.cvtColor(
                (rgb_curr.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8),
                cv2.COLOR_RGB2GRAY,
            )
            if self.mode != TemporalMode.ADAIN_EMA
            else None
        )

        if self.mode == TemporalMode.ADAIN_EMA:
            if self._prev_rgb is None:
                self._prev_rgb = rgb_curr.detach()
                return stylized_norm
            blended = self.ema_beta * rgb_curr + (1 - self.ema_beta) * self._prev_rgb
            self._prev_rgb = blended.detach()
            mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
            out_norm = (blended - mean) / std
            return out_norm

        assert gray is not None

        if self._prev_rgb is None or self._prev_gray is None:
            self._prev_rgb = rgb_curr.detach()
            self._prev_gray = gray
            return stylized_norm

        flow_fwd = optical_flow_farneback(self._prev_gray, gray)
        flow_bwd = optical_flow_farneback(gray, self._prev_gray)
        occ = compute_occlusion_mask(flow_fwd, flow_bwd)
        grid = flow_to_grid(flow_fwd, self.device)
        prev_rgb = self._prev_rgb.to(self.device)
        warped_prev = warp_tensor_with_flow(prev_rgb, grid)
        occ_t = torch.from_numpy(occ).view(1, 1, occ.shape[0], occ.shape[1]).to(self.device)
        blended_rgb = blend_with_previous_flow(rgb_curr, warped_prev, self.flow_blend_beta, occ_t)

        self._prev_rgb = blended_rgb.detach()
        self._prev_gray = gray
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        return (blended_rgb - mean) / std
