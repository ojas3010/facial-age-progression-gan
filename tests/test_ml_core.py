"""Unit tests for the ML core, mirroring the validation matrix in the
project report (Table 8.1.1)."""
import torch
from PIL import Image
import numpy as np

from ml_core.dataset import get_age_group, get_loaders, UTKFaceDataset
from ml_core.inference import label2onehot, AgeProgressor
from ml_core.model import Generator, Discriminator


def test_get_age_group_lower_boundaries():
    assert [get_age_group(a) for a in (0, 10, 11)] == [0, 0, 1]


def test_get_age_group_upper_boundaries():
    assert [get_age_group(a) for a in (50, 51, 100)] == [4, 5, 5]


def test_dataset_parses_valid_filenames(tmp_path):
    Image.new("RGB", (64, 64)).save(tmp_path / "25_0_0_20170116174525125.jpg")
    Image.new("RGB", (64, 64)).save(tmp_path / "70_1_2_20170116174525126.jpg")
    ds = UTKFaceDataset(str(tmp_path))
    assert len(ds) == 2
    assert sorted(ds.labels) == [2, 5]


def test_dataset_skips_corrupted_filenames(tmp_path):
    Image.new("RGB", (64, 64)).save(tmp_path / "25_0_0_20170116174525125.jpg")
    Image.new("RGB", (64, 64)).save(tmp_path / "notanage_junk.jpg")
    Image.new("RGB", (64, 64)).save(tmp_path / "garbage.jpg")
    ds = UTKFaceDataset(str(tmp_path))
    assert len(ds) == 1


def test_train_val_split_is_deterministic_and_disjoint(tmp_path):
    for i in range(10):
        Image.new("RGB", (64, 64)).save(tmp_path / f"{20 + i}_0_0_2017{i:04d}.jpg")
    train_loader, val_loader = get_loaders(str(tmp_path), image_size=64, batch_size=2, num_workers=0)
    train_idx = set(train_loader.dataset.indices)
    val_idx = set(val_loader.dataset.indices)
    assert len(train_idx) == 8 and len(val_idx) == 2
    assert train_idx.isdisjoint(val_idx)
    # Same seed, same directory: split must be identical on a rebuild
    _, val_loader2 = get_loaders(str(tmp_path), image_size=64, batch_size=2, num_workers=0)
    assert set(val_loader2.dataset.indices) == val_idx


def test_label2onehot():
    labels = torch.tensor([0, 3, 5])
    onehot = label2onehot(labels, 6)
    assert onehot.shape == (3, 6)
    assert onehot.sum().item() == 3
    assert onehot[0, 0] == 1 and onehot[1, 3] == 1 and onehot[2, 5] == 1


def test_generator_output_shape_and_bounds():
    g = Generator(c_dim=6, repeat_num=6)
    g.eval()
    x = torch.randn(1, 3, 128, 128)
    c = label2onehot(torch.tensor([2]), 6)
    with torch.no_grad():
        out = g(x, c)
    assert out.shape == (1, 3, 128, 128)
    assert out.min() >= -1.0 and out.max() <= 1.0


def test_discriminator_output_shapes():
    d = Discriminator(image_size=128, c_dim=6, repeat_num=6)
    d.eval()
    with torch.no_grad():
        out_src, out_cls = d(torch.randn(1, 3, 128, 128))
    assert out_src.shape == (1, 1, 2, 2)
    assert out_cls.shape == (1, 6)


def test_progress_age_end_to_end():
    progressor = AgeProgressor()  # uninitialized weights: output is noise but shapes must hold
    dummy = Image.fromarray(np.uint8(np.random.rand(256, 256, 3) * 255))
    out = progressor.progress_age(dummy, 3)
    assert isinstance(out, Image.Image)
    assert out.size == (128, 128)
