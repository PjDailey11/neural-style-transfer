"""Classical Gatys et al. style transfer backbone (VGG-19 + average pooling)."""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterable

import torch
import torch.nn as nn
from torchvision import models

from nst.utils.gram import gram_matrix

# Indices of ReLU outputs in torchvision VGG-19 ``features`` after pool substitution.
VGG19_RELU_INDEX_TO_NAME: dict[int, str] = {
    1: "relu1_1",
    3: "relu1_2",
    6: "relu2_1",
    8: "relu2_2",
    11: "relu3_1",
    13: "relu3_2",
    15: "relu3_3",
    17: "relu3_4",
    20: "relu4_1",
    22: "relu4_2",
    24: "relu4_3",
    26: "relu4_4",
    29: "relu5_1",
    31: "relu5_2",
    33: "relu5_3",
    35: "relu5_4",
}


def _replace_maxpool_with_avgpool(features: nn.Sequential) -> nn.Sequential:
    """Swap MaxPool for AveragePool as recommended in Gatys-style NST."""
    modules = []
    for layer in features.children():
        if isinstance(layer, nn.MaxPool2d):
            modules.append(nn.AvgPool2d(kernel_size=2, stride=2))
        else:
            modules.append(layer)
    return nn.Sequential(*modules)


class GatysVGG(nn.Module):
    """Frozen VGG-19 feature extractor with avg pooling and named ReLU outputs."""

    default_style_layers: tuple[str, ...] = ("relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1")
    default_content_layer: str = "relu4_2"

    def __init__(
        self,
        style_layers: Iterable[str] | None = None,
        content_layer: str | None = None,
        weights: models.VGG19_Weights | None = models.VGG19_Weights.IMAGENET1K_V1,
    ) -> None:
        super().__init__()
        sty = tuple(style_layers) if style_layers is not None else self.default_style_layers
        self.style_layer_names = sty
        self.content_layer_name = content_layer or self.default_content_layer

        backbone = models.vgg19(weights=weights).features
        backbone = _replace_maxpool_with_avgpool(backbone)
        full = copy.deepcopy(backbone).eval()
        for p in full.parameters():
            p.requires_grad_(False)

        needed = set(sty) | {self.content_layer_name}
        last_idx = max(i for i, n in VGG19_RELU_INDEX_TO_NAME.items() if n in needed)
        trimmed = list(full.children())[: last_idx + 1]
        self.features = nn.Sequential(*trimmed)

        self.layer_weights: dict[str, float] = {name: 1.0 for name in sty}

    def forward(self, x: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Returns style activations dict and content activation tensor."""
        acts: dict[str, torch.Tensor] = {}
        content_feat: torch.Tensor | None = None
        h = x
        for idx, layer in enumerate(self.features.children()):
            h = layer(h)
            name = VGG19_RELU_INDEX_TO_NAME.get(idx)
            if name is None:
                continue
            if name in self.style_layer_names:
                acts[name] = h
            if name == self.content_layer_name:
                content_feat = h
        if content_feat is None:
            raise RuntimeError(f"Content layer {self.content_layer_name} not reached in forward pass.")
        return acts, content_feat


def run_gatys_adam(
    device: torch.device,
    content: torch.Tensor,
    style: torch.Tensor,
    model: GatysVGG,
    steps: int = 300,
    alpha_content: float = 1.0,
    beta_style: float = 1e6,
    lr: float = 1.0,
    log_every: int = 50,
    step_callback: Callable[[int, int, float], None] | None = None,
    callback_every: int = 1,
) -> torch.Tensor:
    """Optimize output image with Adam (default) — LBFGS optional via caller.

    Note: see ``run_gatys_lbfgs`` below; Adam is often more robust on GPU for batch size 1.

    Args:
        content: Normalized content batch ``(1,3,H,W)``.
        style: Normalized style batch ``(1,3,H,W)``.
        model: Frozen ``GatysVGG``.
        steps: Optimization iterations.
        alpha_content: Weight on content loss (α).
        beta_style: Weight on style loss (β).
        lr: Learning rate for Adam fallback path.
        step_callback: Optional hook invoked as ``(step, total_steps, loss)`` with *step*
            in ``1 .. total_steps``. Useful for UIs (e.g. Streamlit progress).
        callback_every: Invoke ``step_callback`` every N steps (and always on the final step).

    Returns:
        Stylized tensor in normalized ImageNet space.
    """
    style_feats, _ = model(style)
    _, content_feat = model(content)

    gen = content.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([gen], lr=lr)

    mse = torch.nn.MSELoss()

    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        gen_acts, gen_content = model(gen)

        loss_c = mse(gen_content, content_feat)

        loss_s = torch.zeros((), device=device, dtype=gen.dtype)
        for name in model.style_layer_names:
            g = gram_matrix(gen_acts[name])
            s = gram_matrix(style_feats[name].detach())
            loss_s = loss_s + mse(g, s)

        loss = alpha_content * loss_c + beta_style * loss_s
        loss.backward()
        opt.step()
        with torch.no_grad():
            mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
            gen.clamp_(-mean / std, (1 - mean) / std)

        if log_every and (step + 1) % log_every == 0:
            print(f"[gatys-adam] step {step+1}/{steps} loss={loss.item():.4f}")

        if step_callback is not None and (callback_every <= 1 or (step + 1) % callback_every == 0 or step == steps - 1):
            step_callback(step + 1, steps, float(loss.detach()))

    return gen.detach()


def run_gatys_lbfgs(
    device: torch.device,
    content: torch.Tensor,
    style: torch.Tensor,
    model: GatysVGG,
    max_iter: int = 50,
    alpha_content: float = 1.0,
    beta_style: float = 1e6,
) -> torch.Tensor:
    """LBFGS optimization as in Gatys et al. (second-order, fewer outer iterations)."""
    style_feats, _ = model(style)
    _, content_feat = model(content)

    gen = content.clone().detach()
    gen.requires_grad_(True)

    optimizer = torch.optim.LBFGS([gen], max_iter=max_iter)
    mse = torch.nn.MSELoss()
    step_state = {"n": 0}

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        gen_acts, gen_content = model(gen)
        loss_c = mse(gen_content, content_feat)
        loss_s = torch.zeros((), device=device, dtype=gen.dtype)
        for name in model.style_layer_names:
            g = gram_matrix(gen_acts[name])
            s = gram_matrix(style_feats[name].detach())
            loss_s = loss_s + mse(g, s)
        loss = alpha_content * loss_c + beta_style * loss_s
        loss.backward()
        step_state["n"] += 1
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        gen.clamp_(-mean / std, (1 - mean) / std)
    return gen.detach()
