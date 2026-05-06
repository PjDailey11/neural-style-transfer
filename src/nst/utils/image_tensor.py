"""Image loading/saving as normalized tensors for VGG-based pipelines."""

from __future__ import annotations

from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_image_tensor(
    path: str | Path,
    image_size: int | None = None,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, tuple[int, int]]:
    """Load an RGB image as ImageNet-normalized CHW tensor ``(1,3,H,W)``.

    Args:
        path: Filesystem path.
        image_size: If set, resize so smaller edge equals ``image_size`` (maintains aspect).
        device: Optional device for output tensor.

    Returns:
        Tuple of tensor and ``(height, width)`` before normalization (after resize).
    """
    img = Image.open(path).convert("RGB")
    w, h = img.size
    transforms: list = []
    if image_size is not None:
        transforms.append(T.Resize(image_size))
    transforms.extend([T.ToTensor()])
    to_tensor = T.Compose(transforms)
    t = to_tensor(img).unsqueeze(0)
    _, _, oh, ow = t.shape
    mean = IMAGENET_MEAN.to(t.device)
    std = IMAGENET_STD.to(t.device)
    t = (t - mean) / std
    if device is not None:
        t = t.to(device)
    return t, (oh, ow)


def rgb01_to_imagenet(batch_01: torch.Tensor) -> torch.Tensor:
    """Map RGB in ``[0,1]`` to ImageNet-normalized tensors."""
    mean = IMAGENET_MEAN.to(batch_01.device, dtype=batch_01.dtype)
    std = IMAGENET_STD.to(batch_01.device, dtype=batch_01.dtype)
    return (batch_01 - mean) / std


def denormalize_imagenet(batch: torch.Tensor) -> torch.Tensor:
    """Invert ImageNet normalization for saving preview frames."""
    mean = IMAGENET_MEAN.to(batch.device, dtype=batch.dtype)
    std = IMAGENET_STD.to(batch.device, dtype=batch.dtype)
    return (batch * std + mean).clamp(0.0, 1.0)


def save_image_tensor(batch: torch.Tensor, path: str | Path) -> None:
    """Save a single-image batch ``(1,3,H,W)`` in ``[0,1]`` space to disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    x = batch.detach().cpu().clamp(0, 1).squeeze(0)
    arr = (x.numpy().transpose(1, 2, 0) * 255.0).round().astype("uint8")
    Image.fromarray(arr).save(path)
