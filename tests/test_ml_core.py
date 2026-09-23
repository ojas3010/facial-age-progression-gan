"""Unit tests for the ML core, mirroring the validation matrix in the
project report (Table 8.1.1)."""
import pytest
import torch
from PIL import Image
import numpy as np

from ml_core.dataset import get_age_group, get_loaders, UTKFaceDataset
from ml_core.inference import AgeProgressor
from ml_core.model import Generator, Discriminator, generator_state_dict, label2onehot


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


def test_get_loaders_rejects_empty_dataset(tmp_path):
    # Must name the directory, not die later in the DataLoader sampler
    with pytest.raises(ValueError, match="No valid images"):
        get_loaders(str(tmp_path / "missing"), num_workers=0)


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


def test_generator_eval_mode_matches_train_mode():
    # The backend serves G in eval(); training and the sample grids run it in
    # train(). Both must compute the same function, however many training
    # forward passes came first.
    torch.manual_seed(0)
    g = Generator(c_dim=6, repeat_num=6)
    with torch.no_grad():
        for _ in range(3):
            g(torch.randn(2, 3, 128, 128), label2onehot(torch.tensor([1, 4]), 6))
        x = torch.randn(1, 3, 128, 128)
        c = label2onehot(torch.tensor([3]), 6)
        out_train = g.train()(x, c)
        out_eval = g.eval()(x, c)
    assert torch.allclose(out_train, out_eval, atol=1e-6)


def legacy_checkpoint(g):
    """G weights as saved before InstanceNorm running stats were disabled,
    wrapped the way DataParallel/Lightning-style checkpoints are."""
    state = dict(g.state_dict())
    for name, module in g.named_modules():
        if isinstance(module, torch.nn.InstanceNorm2d):
            state[f"{name}.running_mean"] = torch.zeros(module.num_features)
            state[f"{name}.running_var"] = torch.ones(module.num_features)
            state[f"{name}.num_batches_tracked"] = torch.tensor(100)
    return {"state_dict": {f"module.{k}": v for k, v in state.items()}}


def test_generator_state_dict_loads_legacy_checkpoint_strictly():
    trained = Generator(c_dim=6, repeat_num=6)
    fresh = Generator(c_dim=6, repeat_num=6)
    fresh.load_state_dict(generator_state_dict(legacy_checkpoint(trained)))  # strict
    for (name, a), b in zip(trained.state_dict().items(), fresh.state_dict().values()):
        assert torch.equal(a, b), name


def test_progressor_loads_legacy_checkpoint_file(tmp_path):
    trained = Generator(c_dim=6, repeat_num=6)
    path = tmp_path / "latest-G.ckpt"
    torch.save(legacy_checkpoint(trained), path)
    progressor = AgeProgressor(model_path=str(path))
    assert progressor.weights_loaded
    first = next(iter(trained.state_dict()))
    assert torch.equal(progressor.G.state_dict()[first].cpu(), trained.state_dict()[first])


def test_progressor_flags_mismatched_checkpoint(tmp_path):
    # A D checkpoint in G's place shares no keys with the generator. Loading
    # non-strictly would "succeed" and serve random weights as if trained.
    path = tmp_path / "latest-G.ckpt"
    torch.save(Discriminator(image_size=128, c_dim=6, repeat_num=6).state_dict(), path)
    assert not AgeProgressor(model_path=str(path)).weights_loaded


def test_preprocessing_center_crops_instead_of_squashing():
    # 256x128: red outer quarters, blue center square. Squashing to 128x128
    # would keep the red bands; a center crop keeps only the blue square.
    img = Image.new("RGB", (256, 128), (255, 0, 0))
    img.paste((0, 0, 255), (64, 0, 192, 128))
    x = AgeProgressor().transform(img)
    assert x.shape == (3, 128, 128)
    assert x[0].max() < -0.9 and x[2].min() > 0.9  # no red anywhere, all blue
