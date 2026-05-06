#!/usr/bin/env python3
"""Temporal video stylization CLI."""

from __future__ import annotations

import argparse

import cv2
import numpy as np
import torch

from nst.models.adain_net import AdaINNet
from nst.models.reconet_style import ReCoNetStyle
from nst.models.temporal_video import TemporalMode, TemporalStylizationPipeline
from nst.utils.image_tensor import denormalize_imagenet, rgb01_to_imagenet


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stylize a video with temporal coherence.")
    p.add_argument("--input", required=True, type=str)
    p.add_argument("--style", required=True, type=str)
    p.add_argument("--output", required=True, type=str)
    p.add_argument(
        "--mode",
        choices=[m.value for m in TemporalMode],
        default=TemporalMode.ADAIN_FLOW.value,
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--adain-checkpoint", dest="adain_checkpoint", type=str, default=None)
    p.add_argument("--reconet-checkpoint", dest="reconet_checkpoint", type=str, default=None)
    p.add_argument("--ema-beta", dest="ema_beta", type=float, default=0.85)
    p.add_argument("--flow-beta", dest="flow_beta", type=float, default=0.65)
    p.add_argument("--max-frames", dest="max_frames", type=int, default=None)
    return p.parse_args()


def tensor_from_bgr(frame_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    chw = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)
    return rgb01_to_imagenet(chw)


def write_frame(writer: cv2.VideoWriter, tensor_norm: torch.Tensor) -> None:
    rgb = denormalize_imagenet(tensor_norm.detach().cpu()).clamp(0, 1).squeeze(0).numpy().transpose(1, 2, 0)
    bgr = (rgb * 255.0).astype(np.uint8)
    bgr = cv2.cvtColor(bgr, cv2.COLOR_RGB2BGR)
    writer.write(bgr)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video {args.input}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(args.output), fourcc, fps, (width, height))

    mode = TemporalMode(args.mode)
    adain: AdaINNet | None = None
    reconet: ReCoNetStyle | None = None

    if mode in (TemporalMode.ADAIN_EMA, TemporalMode.ADAIN_FLOW):
        adain = AdaINNet().to(device)
        if args.adain_checkpoint:
            state = torch.load(args.adain_checkpoint, map_location=device)
            adain.decoder.load_state_dict(state)
    elif mode == TemporalMode.RECONET_FLOW:
        reconet = ReCoNetStyle().to(device)
        if args.reconet_checkpoint:
            reconet.load_state_dict(torch.load(args.reconet_checkpoint, map_location=device))

    style_bgr = cv2.imread(str(args.style))
    if style_bgr is None:
        raise SystemExit(f"Cannot read style image: {args.style}")
    style_tensor = tensor_from_bgr(style_bgr, device)
    pipe = TemporalStylizationPipeline(
        mode,
        device,
        adain=adain,
        reconet=reconet,
        ema_beta=args.ema_beta,
        flow_blend_beta=args.flow_beta,
    )

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.max_frames is not None and idx >= args.max_frames:
            break
        frame_t = tensor_from_bgr(frame, device)
        out = pipe.process_frame(frame_t, style_tensor)
        write_frame(writer, out)
        idx += 1

    cap.release()
    writer.release()
    print(f"Wrote {idx} frames to {args.output}")


if __name__ == "__main__":
    main()
