"""Optical flow helpers for temporal coherence (Farneback + occlusion proxies)."""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def optical_flow_farneback(
    gray_prev: np.ndarray,
    gray_curr: np.ndarray,
    pyr_scale: float = 0.5,
    levels: int = 3,
    winsize: int = 15,
    iterations: int = 3,
    poly_n: int = 5,
    poly_sigma: float = 1.2,
) -> np.ndarray:
    """Dense optical flow from OpenCV Farneback (H, W, 2) with vectors (dx, dy)."""
    flow = cv2.calcOpticalFlowFarneback(
        gray_prev,
        gray_curr,
        None,
        pyr_scale,
        levels,
        winsize,
        iterations,
        poly_n,
        poly_sigma,
        0,
    )
    return flow


def flow_to_grid(flow_hw2: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert OpenCV flow to ``grid_sample`` normalized coordinates ``(1,H,W,2)``.

    OpenCV returns displacement in pixel units from prev -> curr.
    To warp previous image toward current: sample prev at ``p - flow(p)``.
    """
    h, w = flow_hw2.shape[:2]
    # grid_sample expects x,y order with normalized coords in [-1,1]
    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    dx = flow_hw2[..., 0]
    dy = flow_hw2[..., 1]
    map_x = xs - dx
    map_y = ys - dy
    gx = (map_x / (w - 1)) * 2 - 1
    gy = (map_y / (h - 1)) * 2 - 1
    grid = np.stack([gx, gy], axis=-1).astype(np.float32)
    t = torch.from_numpy(grid).unsqueeze(0).to(device)
    return t


def warp_tensor_with_flow(tensor_chw: torch.Tensor, grid_nhw2: torch.Tensor) -> torch.Tensor:
    """Warp ``(1,C,H,W)`` tensor using normalized grid from ``flow_to_grid``."""
    return F.grid_sample(tensor_chw, grid_nhw2, mode="bilinear", padding_mode="border", align_corners=True)


def compute_occlusion_mask(
    flow_fwd: np.ndarray,
    flow_bwd: np.ndarray,
    thresh: float = 1.5,
) -> np.ndarray:
    """Forward–backward consistency occlusion proxy.

    Returns weight map ``(H,W)`` in ``[0,1]`` — higher means more reliable warp.

    Args:
        flow_fwd: Forward flow prev -> curr.
        flow_bwd: Backward flow curr -> prev.
        thresh: L2 threshold in pixels for marking unreliable regions.
    """
    h, w = flow_fwd.shape[:2]
    grid_y, grid_x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    fx = grid_x + flow_fwd[..., 0]
    fy = grid_y + flow_fwd[..., 1]
    fx = np.clip(np.round(fx), 0, w - 1).astype(np.int32)
    fy = np.clip(np.round(fy), 0, h - 1).astype(np.int32)
    flow_at_tgt = flow_bwd[fy, fx]
    diff = np.linalg.norm(flow_fwd + flow_at_tgt, axis=-1)
    reliable = (diff < thresh).astype(np.float32)
    return reliable


def blend_with_previous_flow(
    current_rgb01: torch.Tensor,
    previous_rgb01: torch.Tensor,
    blend_beta: float = 0.85,
    occlusion: torch.Tensor | None = None,
) -> torch.Tensor:
    """Blend current stylized frame with temporally warped previous frame.

    Args:
        current_rgb01: ``(1,3,H,W)``
        previous_rgb01: ``(1,3,H,W)``
        blend_beta: Weight on ``current`` vs warped previous ``1-beta``.
        occlusion: Optional ``(1,1,H,W)`` weights for adaptive blending.
    """
    if occlusion is None:
        return blend_beta * current_rgb01 + (1 - blend_beta) * previous_rgb01
    occ = occlusion.clamp(0, 1)
    beta_map = blend_beta * occ + (1 - occ) * 1.0
    return beta_map * current_rgb01 + (1 - beta_map) * previous_rgb01
