# Neural Style Transfer Toolkit

Production-oriented PyTorch implementations of:

1. **Classical Gatys et al. optimization** — VGG-19 with average pooling, Gram-matrix style loss, content/style weighting (α/β), and **Adam** or **L-BFGS** solvers.
2. **Fast AdaIN** — Encoder–decoder feed-forward network with adaptive instance normalization for arbitrary styles.
3. **Video stylization** — AdaIN with exponential moving-average smoothing or optical-flow–guided blending; optional **ReCoNet-style** residual network trained with occlusion-aware temporal loss.

## Layout

- `src/nst/models/` — Gatys backbone, AdaIN, ReCoNet-style residual net, temporal pipelines.
- `src/nst/utils/` — Gram matrices, losses (including temporal Charbonnier term), optical flow helpers, tensor IO.
- `train.py`, `evaluate.py`, `transform_image.py`, `transform_video.py` — CLI entrypoints.
- `streamlit_app.py` — Interactive demos for images (and a lightweight temporal preview).

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

# Gatys (slow, high quality)
python transform_image.py --content ./examples/content.jpg --style ./examples/style.jpg \
  --output ./out_gatys.png --method gatys_adam --steps 300

# AdaIN (requires trained decoder for best results)
python transform_image.py --content ./examples/content.jpg --style ./examples/style.jpg \
  --output ./out_adain.png --method adain --adain-checkpoint ./checkpoints/adain_decoder.pt

# Video + temporal coherence
python transform_video.py --input ./clips/input.mp4 --style ./examples/style.jpg \
  --output ./out_video.mp4 --mode adain_flow

# Streamlit UI
streamlit run streamlit_app.py
```

Training expects ImageFolder layouts (`./data/content/<label>/*.jpg`, `./data/style/<label>/*.jpg`). ReCoNet temporal training consumes consecutive frames inside `./data/frames/*.jpg|png`.

## Docker

```bash
docker build -t nst:latest .
docker run --rm -p 8501:8501 nst:latest
```

## License

MIT.
