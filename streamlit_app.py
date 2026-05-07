#!/usr/bin/env python3
"""Streamlit UI for uploading images/video and selecting stylization modes."""

from __future__ import annotations

import time
from pathlib import Path

import streamlit as st
import torch

from nst.models.adain_net import AdaINNet
from nst.models.gatys_vgg import GatysVGG, run_gatys_adam
from nst.models.temporal_video import TemporalMode, TemporalStylizationPipeline
from nst.utils.image_tensor import denormalize_imagenet, load_image_tensor, save_image_tensor


def main() -> None:
    st.set_page_config(page_title="Neural Style Transfer Lab", layout="wide")
    st.title("Neural Style Transfer Lab")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.caption(f"Device: {device}")

    mode = st.radio(
        "Pipeline",
        ["Gatys (optimization)", "AdaIN (feed-forward)", "Video temporal preview"],
        horizontal=True,
    )

    content_file = st.file_uploader("Content image or video frame", type=["png", "jpg", "jpeg"])
    style_file = st.file_uploader("Style reference", type=["png", "jpg", "jpeg"])

    ckpt = st.text_input("Optional AdaIN decoder checkpoint (.pt)", value="")

    if content_file and style_file:
        tmp_c = Path(".streamlit_cache/content_upload.png")
        tmp_s = Path(".streamlit_cache/style_upload.png")
        tmp_c.parent.mkdir(parents=True, exist_ok=True)
        tmp_c.write_bytes(content_file.getbuffer())
        tmp_s.write_bytes(style_file.getbuffer())

        if mode.startswith("Gatys"):
            if device.type == "cpu":
                st.warning(
                    "You are on **CPU**. Gatys runs many VGG forward/backward passes — "
                    "full‑resolution images can take **many minutes**. "
                    "Lower **Max side** and/or **Adam steps** for a quick preview."
                )
            max_side = st.slider(
                "Gatys: max side length (px)",
                min_value=256,
                max_value=2048,
                value=512,
                step=32,
                help="Resize so the smaller edge matches this value (keeps aspect ratio). " "Smaller is much faster.",
            )
            content, _ = load_image_tensor(tmp_c, image_size=max_side, device=device)
            style, _ = load_image_tensor(tmp_s, image_size=max_side, device=device)

            alpha = st.slider("α content weight", 1e-4, 10.0, 1.0)
            beta = st.slider("β style weight", 1e3, 1e8, 1e6, step=1e5)
            steps = st.slider("Adam steps", 50, 600, 200)

            if st.button("Run Gatys"):
                prog = st.progress(0.0)
                status = st.empty()
                t0 = time.perf_counter()

                def on_step(cur: int, total: int, loss_val: float) -> None:
                    frac = min(cur / total, 1.0)
                    prog.progress(frac)
                    elapsed = time.perf_counter() - t0
                    eta = ""
                    if cur > 0 and elapsed > 0:
                        rate = elapsed / cur
                        rem = rate * (total - cur)
                        eta = f" · ETA ~**{rem:.0f}s**"
                    status.markdown(
                        f"Step **{cur}** / **{total}** · loss **{loss_val:.4f}** · " f"elapsed **{elapsed:.1f}s**{eta}"
                    )

                with st.spinner("Loading VGG-19 (first run may download ~550MB weights)…"):
                    model = GatysVGG().to(device)

                try:
                    out = run_gatys_adam(
                        device,
                        content,
                        style,
                        model,
                        steps=steps,
                        alpha_content=alpha,
                        beta_style=beta,
                        step_callback=on_step,
                        callback_every=max(1, steps // 60),
                    )
                finally:
                    prog.progress(1.0)

                status.markdown(
                    f"**Done** in **{time.perf_counter() - t0:.1f}s** · "
                    f"output size **{tuple(out.shape[-2:])}** (H×W)."
                )
                out_path = Path(".streamlit_cache/out_gatys.png")
                save_image_tensor(denormalize_imagenet(out.cpu()), out_path)
                st.image(str(out_path), caption="Stylized output", use_container_width=True)

        elif mode.startswith("AdaIN"):
            content, _ = load_image_tensor(tmp_c, device=device)
            style, _ = load_image_tensor(tmp_s, device=device)
            if st.button("Run AdaIN"):
                net = AdaINNet().to(device)
                if ckpt and Path(ckpt).exists():
                    net.decoder.load_state_dict(torch.load(ckpt, map_location=device))
                with st.spinner("Stylizing..."):
                    out = net.stylize(content, style)
                    out_path = Path(".streamlit_cache/out_adain.png")
                    save_image_tensor(denormalize_imagenet(out.cpu()), out_path)
                    st.image(str(out_path), caption="AdaIN output", use_container_width=True)
        else:
            content, _ = load_image_tensor(tmp_c, device=device)
            style, _ = load_image_tensor(tmp_s, device=device)
            st.info("Temporal preview: AdaIN plus flow blending on the still image (demo only).")
            tmode = st.selectbox("Temporal mode", [TemporalMode.ADAIN_EMA.value, TemporalMode.ADAIN_FLOW.value])
            if st.button("Simulate two-frame temporal blend"):
                net = AdaINNet().to(device)
                if ckpt and Path(ckpt).exists():
                    net.decoder.load_state_dict(torch.load(ckpt, map_location=device))
                pipe = TemporalStylizationPipeline(
                    TemporalMode(tmode),
                    device,
                    adain=net,
                    ema_beta=0.85,
                    flow_blend_beta=0.65,
                )
                first = pipe.process_frame(content, style)
                second = pipe.process_frame(content * 0.98 + style * 0.02, style)
                c1, c2 = st.columns(2)
                with c1:
                    p1 = Path(".streamlit_cache/temporal1.png")
                    save_image_tensor(denormalize_imagenet(first.cpu()), p1)
                    st.image(str(p1), caption="Frame 1", use_container_width=True)
                with c2:
                    p2 = Path(".streamlit_cache/temporal2.png")
                    save_image_tensor(denormalize_imagenet(second.cpu()), p2)
                    st.image(str(p2), caption="Frame 2 (perturbed input)", use_container_width=True)


if __name__ == "__main__":
    main()
