# DEVLOG — Neural Style Transfer

This log records **what changed**, **why**, and **next steps** after each major milestone.

---

## 2026-05-06 — Initial scaffold

**What changed**

- Added `pyproject.toml` (editable install, runtime + `[dev]` extras, Black/Ruff at 120 cols, pytest config).
- Created package skeleton under `src/nst/` with `models/` and `utils/` (public imports: `nst.models`, `nst.utils`).  
  _Note:_ This mirrors a `/src/models` layout while keeping a proper Python package name (`nst`) instead of importing a generic top-level `models` module.

**Why**

- Single installable surface for CLI, Streamlit, Docker, and CI; avoids name clashes with PyPI modules named `models`.

**Next steps**

- Flesh out Gatys optimization, AdaIN training, and temporal video utilities.

---

## 2026-05-06 — Core algorithms (Gatys, AdaIN, temporal video)

**What changed**

- **Gatys et al.:** `GatysVGG` uses torchvision VGG-19 with **MaxPool → AvgPool** substitution, configurable style/content layers, Gram-based style loss helpers, **Adam** (`run_gatys_adam`) and **L-BFGS** (`run_gatys_lbfgs`) solvers, plus pixel clamping in normalized space.
- **Fast NST:** `AdaINNet` stacks frozen `AdaINEncoder` (through `relu4_1`), AdaIN statistics merge, and a mirror decoder; outputs pass through **sigmoid → RGB → ImageNet normalization** for stable feed-forward training/inference. Optional `encoder_weights=None` for tests/offline smoke runs.
- **Video:** `TemporalStylizationPipeline` supports AdaIN + **EMA smoothing**, AdaIN + **Farneback flow** with forward–backward occlusion weights, and **ReCoNet-style** residuals (`ReCoNetStyle`) with optional `--reconet-checkpoint`. Training pairs frames with **occlusion-aware temporal loss** (`reconet_temporal_loss`).
- Utilities: Gram matrices, perceptual/AdaIN training losses, optical-flow IO and blending helpers, tensor IO helpers (`rgb01_to_imagenet`, etc.).

**Why**

- Matches the three requested pillars (classical optimization NST, AdaIN feed-forward arbitrary style, coherent video via temporal losses / flow).

**Next steps**

- Ship CLI + UI wrappers, container, CI, and regression tests.

---

## 2026-05-06 — CLI, Streamlit, Docker, CI/CD, tests

**What changed**

- Scripts: `train.py` (AdaIN ImageFolder trainer + ReCoNet paired-frame trainer), `evaluate.py` (PSNR), `transform_image.py`, `transform_video.py`.
- `streamlit_app.py` image demos + lightweight temporal simulation.
- `Dockerfile` / `.dockerignore` — slim Python 3.11 image running Streamlit on port 8501.
- `.github/workflows/ci.yml` — Black, Ruff, pytest, Docker Buildx with GHCR login/push on `main`/`master` pushes.
- `tests/` covering Gram shapes, Gatys/AdaIN/ReCoNet forwards, temporal pipeline smoke cases (random weights to avoid CI downloads).
- `README.md`, `.gitignore`, registry entry `neural-style-transfer` in `~/.claude/workspaces/registry.json`.

**Why**

- Operationalizes training/evaluation/deploy loops and guards regressions in CI.

**Next steps**

- Curate downloadable AdaIN/ReCoNet checkpoints and example assets under `examples/` / `data/`.
- Swap Farneback flow for learned flow (e.g., RAFT) if higher fidelity is required at runtime.
- Integrate perceptual metrics (LPIPS) and profiling hooks in `evaluate.py`.
- Wire Streamlit to optional GPU inference queues for concurrent uploads.

---

## 2026-05-06 — Vercel project hub (`web/`)

**What changed**

- Added a minimal **Next.js 14** App Router site under `web/` (static landing + GitHub links). PyTorch/Streamlit remain local/Docker-only by design.
- Bumped Next to **14.2.35** (security-patched line). Documented deploy steps and live URL in `README.md`.
- Deployed to Vercel via CLI; production alias: https://web-six-delta-90.vercel.app (project scope `pj-daileyes-projects`, name `web` — rename in Vercel if desired).

**Why**

- Vercel cannot realistically host Streamlit + Torch inference; the Next hub gives a fast public URL for the repo while keeping compute elsewhere.

**Next steps**

- In Vercel: **Connect Git** → set **Root Directory** to `web` for automatic previews on every push.
- Optional: rename the Vercel project from `web` to `neural-style-transfer` for clarity.
