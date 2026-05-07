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

## Hosting on Vercel

Vercel serves the **Next.js project hub** under `web/` (links, README context). **PyTorch, Streamlit, and GPU inference are not executed on Vercel** — run those locally or via Docker / another GPU host.

**Live site:** https://web-six-delta-90.vercel.app (production alias; inspect deployments in the [Vercel dashboard](https://vercel.com/dashboard)).

### Git → automatic deploys

The repo is connected to Vercel as a **monorepo**: the Next app lives in **`web/`**, while the Python toolkit stays at the root.

**Repo-level `vercel.json`** tells Vercel to:

1. Run **`npm install`** at the repository root (minimal `package.json` declares `next` so version detection works).
2. Run **`npm ci --prefix web`** for real app dependencies.
3. Run **`npm run build --prefix web`** so `next build` sees `web/app`.

You can still set **Root Directory → `web`** in the [Vercel project settings](https://vercel.com/pj-daileyes-projects/web/settings/general) if you prefer; then you may simplify or remove the root `package.json` / `vercel.json` overrides.

CLI deploy from `web/` (optional):

```bash
cd web
npm install
npm run build   # optional local check
npx vercel deploy --prod --yes
```

## License

MIT.
