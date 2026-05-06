#!/usr/bin/env python3
"""Lightweight evaluation helpers (PSNR / basic perceptual distance proxies)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from nst.utils.image_tensor import denormalize_imagenet, load_image_tensor


def psnr(a_01: torch.Tensor, b_01: torch.Tensor, eps: float = 1e-8) -> float:
    """Peak SNR on tensors already scaled to ``[0,1]``."""
    mse = F.mse_loss(a_01, b_01).item()
    if mse < eps:
        return float("inf")
    return float(10.0 * torch.log10(torch.tensor(1.0 / mse)))


def main() -> None:
    p = argparse.ArgumentParser(description="Compare two aligned RGB images.")
    p.add_argument("--reference", required=True, help="Reference image path.")
    p.add_argument("--candidate", required=True, help="Candidate / stylized image path.")
    p.add_argument("--size", type=int, default=None)
    args = p.parse_args()

    ref, _ = load_image_tensor(Path(args.reference), image_size=args.size)
    cand, _ = load_image_tensor(Path(args.candidate), image_size=args.size)
    ref01 = denormalize_imagenet(ref).clamp(0, 1)
    cand01 = denormalize_imagenet(cand).clamp(0, 1)
    score = psnr(ref01, cand01)
    print(f"PSNR (dB): {score:.4f}")


if __name__ == "__main__":
    main()
