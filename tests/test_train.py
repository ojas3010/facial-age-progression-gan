"""Checkpoint bookkeeping in ml_core/train.py: resume discovery, pruning and
atomic saves. File-level only - the training loop itself is exercised by
tests/smoke_train.py."""
import os

import torch

from ml_core.train import find_resume_iter, prune_resume_checkpoints, save_checkpoint


def touch(model_dir, *names):
    for name in names:
        (model_dir / name).write_bytes(b"")


def test_find_resume_iter_on_missing_or_empty_dir(tmp_path):
    assert find_resume_iter(str(tmp_path / "missing")) == 0
    assert find_resume_iter(str(tmp_path)) == 0


def test_find_resume_iter_skips_incomplete_saves_and_stray_files(tmp_path):
    touch(tmp_path, "5-G.ckpt", "5-D.ckpt", "5-opt.ckpt")
    # Save killed after G but before D: restore_model would crash on 10-D
    touch(tmp_path, "10-G.ckpt", "10-opt.ckpt")
    # Files that must not crash int() parsing or count as resume points
    touch(tmp_path, "latest-G.ckpt", "best-G.ckpt", "15-G.ckpt.tmp", "15-D.ckpt.tmp", "notes.txt")
    assert find_resume_iter(str(tmp_path)) == 5


def test_prune_keeps_every_g_and_newest_resume_points(tmp_path):
    for it in (5, 10, 15):
        touch(tmp_path, f"{it}-G.ckpt", f"{it}-D.ckpt", f"{it}-opt.ckpt")
    # Orphan D/opt from a killed save at a later iteration: it is not a
    # resume point, so it must not push 15 out of the keep window
    touch(tmp_path, "20-D.ckpt", "20-opt.ckpt", "latest-G.ckpt")

    prune_resume_checkpoints(str(tmp_path), keep_last=1)

    remaining = set(os.listdir(tmp_path))
    assert {"5-G.ckpt", "10-G.ckpt", "15-G.ckpt", "latest-G.ckpt"} <= remaining
    assert {"15-D.ckpt", "15-opt.ckpt"} <= remaining
    assert not {"5-D.ckpt", "5-opt.ckpt", "10-D.ckpt", "10-opt.ckpt"} & remaining
    assert find_resume_iter(str(tmp_path)) == 15


def test_prune_keep_last_larger_than_history_deletes_nothing(tmp_path):
    for it in (5, 10):
        touch(tmp_path, f"{it}-G.ckpt", f"{it}-D.ckpt", f"{it}-opt.ckpt")
    before = set(os.listdir(tmp_path))
    prune_resume_checkpoints(str(tmp_path), keep_last=3)
    assert set(os.listdir(tmp_path)) == before


def test_save_checkpoint_writes_complete_file_without_temp_leftover(tmp_path):
    path = str(tmp_path / "5-G.ckpt")
    save_checkpoint({"w": torch.arange(4)}, path)
    assert torch.equal(torch.load(path, weights_only=True)["w"], torch.arange(4))
    assert os.listdir(tmp_path) == ["5-G.ckpt"]
    # Overwrite in place (latest-G.ckpt is rewritten every save)
    save_checkpoint({"w": torch.zeros(2)}, path)
    assert torch.equal(torch.load(path, weights_only=True)["w"], torch.zeros(2))
