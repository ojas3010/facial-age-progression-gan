# GAN-Based Facial Age Progression and Regression

Identity-preserving facial age progression/regression using a multi-domain
conditional GAN (StarGAN topology, WGAN-GP objective) trained on UTKFace,
served through a FastAPI backend and a React dashboard.

## Structure

| Path | Purpose |
|------|---------|
| `ml_core/model.py` | StarGAN Generator / Discriminator |
| `ml_core/dataset.py` | UTKFace loader, 6 age domains (0-10 … 51+) |
| `ml_core/train.py` | WGAN-GP training loop with checkpoint resume |
| `ml_core/inference.py` | `AgeProgressor` inference wrapper |
| `backend/main.py` | FastAPI service (port 8000) |
| `frontend/` | React 19 + Vite SPA (dev port 5173) |
| `tests/` | pytest unit + API tests, training smoke test |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt
```

GPU users: install the CUDA build of PyTorch from https://pytorch.org instead
of the default CPU wheel.

## Run

**Backend** (from repo root):
```bash
python -m uvicorn backend.main:app --port 8000
```
Model weights are loaded from `ml_core/models/latest-G.ckpt` at startup.
Without the checkpoint the API still runs but outputs noise.

**Frontend**:
```bash
cd frontend
npm install
npm run dev            # http://localhost:5173
```

## API

`POST /api/progress_age` — multipart form:
- `file`: JPEG/PNG/WEBP image, max 5 MB
- `target_age_group`: integer 0–5 (0-10, 11-20, 21-30, 31-40, 41-50, 51+)

Returns `{"status": "success", "image_base64": "<jpeg>"}`.
Errors: 400 (bad file type / too large / corrupt image), 422 (invalid age group),
500 (inference failure).

## Training

Place UTKFace images (`[age]_[gender]_[race]_[date].jpg`) in
`ml_core/data/utkface`, then:

```bash
cd ml_core
python train.py
```

Checkpoints (`{iter}-G.ckpt`, `{iter}-D.ckpt`, `{iter}-opt.ckpt`) are written to
`ml_core/models` every 1,000 iterations along with the `latest-G.ckpt` alias the
backend consumes. Interrupted runs resume automatically from the latest
checkpoint, including optimizer state.

## Tests

```bash
pip install pytest httpx
python -m pytest tests -v            # unit + API tests
python tests/smoke_train.py          # 10-iteration CPU training smoke test
```
