#!/usr/bin/env python3
"""CLI for single-image neural style transfer."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from nst.models.adain_net import AdaINNet
from nst.models.gatys_vgg import GatysVGG, run_gatys_adam, run_gatys_lbfgs
from nst.utils.image_tensor import denormalize_imagenet, load_image_tensor, save_image_tensor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stylize a single image.")
    p.add_argument("--content", required=True, type=str)
    p.add_argument("--style", required=True, type=str)
    p.add_argument("--output", required=True, type=str)
    p.add_argument(
        "--method",
        choices=["gatys_adam", "gatys_lbfgs", "adain"],
        default="gatys_adam",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--image-size", dest="image_size", type=int, default=None)

    p.add_argument("--alpha-content", dest="alpha_content", type=float, default=1.0)
    p.add_argument("--beta-style", dest="beta_style", type=float, default=1e6)
    p.add_argument("--steps", type=int, default=250)
    p.add_argument("--lbfgs-iters", dest="lbfgs_iters", type=int, default=60)

    p.add_argument("--adain-checkpoint", dest="adain_checkpoint", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    content, _ = load_image_tensor(Path(args.content), image_size=args.image_size, device=device)
    style, _ = load_image_tensor(Path(args.style), image_size=args.image_size, device=device)

    if args.method.startswith("gatys"):
        model = GatysVGG().to(device)
        if args.method == "gatys_lbfgs":
            out = run_gatys_lbfgs(
                device,
                content,
                style,
                model,
                max_iter=args.lbfgs_iters,
                alpha_content=args.alpha_content,
                beta_style=args.beta_style,
            )
        else:
            out = run_gatys_adam(
                device,
                content,
                style,
                model,
                steps=args.steps,
                alpha_content=args.alpha_content,
                beta_style=args.beta_style,
            )
    else:
        net = AdaINNet().to(device)
        if args.adain_checkpoint:
            state = torch.load(args.adain_checkpoint, map_location=device)
            net.decoder.load_state_dict(state)
        out = net.stylize(content, style)

    rgb = denormalize_imagenet(out.cpu())
    save_image_tensor(rgb, Path(args.output))
    print(f"Wrote stylized image to {args.output}")


if __name__ == "__main__":
    main()
