#!/usr/bin/env python3
"""Training entrypoints for AdaIN fast NST and ReCoNet-style video networks."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.datasets.folder import ImageFolder
from tqdm import tqdm

from nst.models.adain_net import AdaINEncoder, AdaINNet
from nst.models.reconet_style import ReCoNetStyle
from nst.utils.flow import compute_occlusion_mask, flow_to_grid, optical_flow_farneback, warp_tensor_with_flow
from nst.utils.image_tensor import denormalize_imagenet
from nst.utils.losses import adain_training_losses, reconet_temporal_loss


class PairedFrameDataset(Dataset):
    """Consecutive frames for occlusion-aware temporal supervision."""

    def __init__(self, frame_dir: Path) -> None:
        paths = sorted(frame_dir.glob("*.jpg")) + sorted(frame_dir.glob("*.png"))
        if len(paths) < 2:
            raise ValueError(f"Need at least two frames in {frame_dir}")
        self.pairs = list(zip(paths[:-1], paths[1:], strict=True))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        from PIL import Image

        p0, p1 = self.pairs[idx]
        to_tensor = T.Compose([T.ToTensor()])
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        i0 = to_tensor(Image.open(p0).convert("RGB")).unsqueeze(0)
        i1 = to_tensor(Image.open(p1).convert("RGB")).unsqueeze(0)
        return ((i0 - mean) / std).squeeze(0), ((i1 - mean) / std).squeeze(0)


def train_adain(args: argparse.Namespace, device: torch.device) -> None:
    content_tf = T.Compose([T.Resize(args.image_size), T.CenterCrop(args.image_size), T.ToTensor()])
    style_tf = T.Compose([T.Resize(args.image_size), T.CenterCrop(args.image_size), T.ToTensor()])

    content_ds = ImageFolder(args.content_dir, transform=content_tf)
    style_ds = ImageFolder(args.style_dir, transform=style_tf)

    content_loader = DataLoader(content_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    style_loader = DataLoader(style_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    model = AdaINNet().to(device)
    opt = torch.optim.Adam(model.decoder.parameters(), lr=args.lr)

    style_iter = iter(style_loader)
    encoder: AdaINEncoder = model.encoder

    content_layers = tuple(args.content_layers.split(","))
    style_layers = tuple(args.style_layers.split(","))

    for epoch in range(args.epochs):
        bar = tqdm(content_loader, desc=f"epoch {epoch+1}/{args.epochs}")
        for batch, _ in bar:
            batch = batch.to(device)
            try:
                style_batch, _ = next(style_iter)
            except StopIteration:
                style_iter = iter(style_loader)
                style_batch, _ = next(style_iter)
            style_batch = style_batch.to(device)

            opt.zero_grad(set_to_none=True)
            out = model.forward_train(batch, style_batch)
            loss, diag = adain_training_losses(
                out,
                batch,
                style_batch,
                encoder,
                content_layers=content_layers,
                style_layers=style_layers,
                lambda_style=args.lambda_style,
            )
            loss.backward()
            opt.step()
            bar.set_postfix(loss=f"{diag['loss_total']:.4f}")

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    ckpt = Path(args.checkpoint_dir) / "adain_decoder.pt"
    torch.save(model.decoder.state_dict(), ckpt)
    print(f"Saved AdaIN decoder weights to {ckpt}")


def train_reconet(args: argparse.Namespace, device: torch.device) -> None:
    import cv2
    import numpy as np

    ds = PairedFrameDataset(Path(args.paired_frames_dir))
    loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)
    model = ReCoNetStyle().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        bar = tqdm(loader, desc=f"reconet epoch {epoch+1}/{args.epochs}")
        for f0, f1 in bar:
            f0 = f0.unsqueeze(0).to(device)
            f1 = f1.unsqueeze(0).to(device)
            rgb0 = denormalize_imagenet(f0)
            rgb1 = denormalize_imagenet(f1)
            g0 = cv2.cvtColor(
                (rgb0.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8),
                cv2.COLOR_RGB2GRAY,
            )
            g1 = cv2.cvtColor(
                (rgb1.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8),
                cv2.COLOR_RGB2GRAY,
            )
            flow = optical_flow_farneback(g0, g1)
            flow_rev = optical_flow_farneback(g1, g0)
            occ = compute_occlusion_mask(flow, flow_rev)

            inp0 = rgb0 * 2.0 - 1.0
            inp1 = rgb1 * 2.0 - 1.0
            out0 = model(inp0)
            out1 = model(inp1)
            rgb_out0 = (out0 + 1.0) * 0.5
            rgb_out1 = (out1 + 1.0) * 0.5

            grid = flow_to_grid(flow, device)
            warped_styl0 = warp_tensor_with_flow(rgb_out0, grid)
            occ_t = torch.from_numpy(occ).view(1, 1, occ.shape[0], occ.shape[1]).to(device)

            pixel = F.l1_loss(rgb_out0, rgb0) + F.l1_loss(rgb_out1, rgb1)
            temp = reconet_temporal_loss(rgb_out1, warped_styl0, occ_t)
            loss = pixel + args.lambda_temp * temp

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            bar.set_postfix(loss=float(loss.detach()))

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    ckpt = Path(args.checkpoint_dir) / "reconet.pt"
    torch.save(model.state_dict(), ckpt)
    print(f"Saved ReCoNet checkpoint to {ckpt}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train neural style transfer models.")
    p.add_argument("--architecture", choices=["adain", "reconet"], required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    p.add_argument("--content-dir", dest="content_dir", type=str, default="./data/content")
    p.add_argument("--style-dir", dest="style_dir", type=str, default="./data/style")
    p.add_argument("--paired-frames-dir", dest="paired_frames_dir", type=str, default="./data/frames")
    p.add_argument("--checkpoint-dir", dest="checkpoint_dir", type=str, default="./checkpoints")

    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=4)
    p.add_argument("--image-size", dest="image_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--lambda-style", dest="lambda_style", type=float, default=1.0)
    p.add_argument("--lambda-temp", dest="lambda_temp", type=float, default=0.5)
    p.add_argument(
        "--content-layers",
        dest="content_layers",
        type=str,
        default="relu4_1,relu3_1",
    )
    p.add_argument(
        "--style-layers",
        dest="style_layers",
        type=str,
        default="relu1_1,relu1_2,relu2_1,relu2_2,relu3_1,relu3_2,relu3_3,relu3_4,relu4_1",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if args.architecture == "adain":
        train_adain(args, device)
    else:
        train_reconet(args, device)


if __name__ == "__main__":
    main()
