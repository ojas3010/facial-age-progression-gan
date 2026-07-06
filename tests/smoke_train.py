"""Manual smoke test for the training loop (not run by pytest: name lacks
test_ prefix). Runs a few CPU iterations on synthetic data and verifies
checkpointing, the latest-G.ckpt alias, and optimizer-state resume.

Usage: .venv/bin/python tests/smoke_train.py
"""
import os
import shutil
import sys
import tempfile

import numpy as np
from PIL import Image

ML_CORE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ml_core")
sys.path.insert(0, ML_CORE)
os.chdir(ML_CORE)

from train import Solver  # noqa: E402

work = tempfile.mkdtemp(prefix="smoke_train_")
data_dir = os.path.join(work, "data")
model_dir = os.path.join(work, "models")
os.makedirs(data_dir)
os.makedirs(model_dir)

# Synthetic "UTKFace" images across several age groups
for i, age in enumerate([5, 15, 25, 35, 45, 70, 8, 60]):
    arr = np.uint8(np.random.rand(128, 128, 3) * 255)
    Image.fromarray(arr).save(os.path.join(data_dir, f"{age}_0_0_2017{i:04d}.jpg"))

config = {
    'c_dim': 6, 'image_size': 128, 'g_repeat_num': 6, 'd_repeat_num': 6,
    'g_lr': 0.0001, 'd_lr': 0.0001, 'beta1': 0.5, 'beta2': 0.999,
    'image_dir': data_dir, 'batch_size': 4, 'num_iters': 5, 'n_critic': 5,
    'lambda_cls': 1, 'lambda_rec': 10, 'lambda_gp': 10,
    'log_step': 5, 'model_save_step': 5, 'model_save_dir': model_dir,
    'num_workers': 0,  # in-process loading: avoids multiprocessing issues in CI/sandboxes
}

print("=== Phase 1: fresh training, 5 iterations ===")
Solver(config).train()

expected = ['5-G.ckpt', '5-D.ckpt', '5-opt.ckpt', 'latest-G.ckpt']
for f in expected:
    path = os.path.join(model_dir, f)
    assert os.path.exists(path), f"missing checkpoint: {f}"
print(f"Checkpoints written: {sorted(os.listdir(model_dir))}")

print("=== Phase 2: resume from iteration 5, run to 10 ===")
config['num_iters'] = 10
solver = Solver(config)
assert solver.get_latest_checkpoint() == 5, "latest-G.ckpt must not confuse checkpoint discovery"
solver.train()
assert os.path.exists(os.path.join(model_dir, '10-G.ckpt'))
print(f"After resume: {sorted(os.listdir(model_dir))}")

shutil.rmtree(work)
print("SMOKE TRAIN OK")
