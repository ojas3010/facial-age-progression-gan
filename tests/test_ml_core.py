"""Unit tests for the ML core, mirroring the validation matrix in the
project report (Table 8.1.1)."""
import math
import os

import cv2
import pytest
import torch
from PIL import Image
import numpy as np

from ml_core.dataset import get_age_group, get_loaders, UTKFaceDataset
from ml_core.inference import AgeProgressor, FaceAligner, NoFaceDetectedError, UTKFACE_TEMPLATE
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


# ----- face detection and alignment -----

UTKFACE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "ml_core", "data", "utkface")
needs_utkface = pytest.mark.skipif(not os.path.isdir(UTKFACE_DIR), reason="UTKFace not present (see README)")


@pytest.fixture(scope="module")
def aligner():
    return FaceAligner(output_size=128)


def place_face(side, angle_deg, center):
    """Template landmarks for a face `side` px wide, rotated by angle_deg
    (image coordinates, y down) about its middle and centred at `center`."""
    a = math.radians(angle_deg)
    rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    return (UTKFACE_TEMPLATE - 0.5) * side @ rot.T + center


def face_row(landmarks, side):
    """A detector row (x, y, w, h, landmarks, score) for place_face output."""
    x, y = landmarks.mean(axis=0) - side / 2
    return [x, y, side, side, *landmarks.ravel(), 0.99]


def paint_eyes(image, landmarks, radius, color):
    for x, y in landmarks[:2]:
        cv2.circle(image, (round(x), round(y)), radius, color, -1)


def test_alignment_levels_eyes_and_fits_template(aligner):
    # A face tilted 25 degrees: after alignment the eyes must be level and
    # every landmark on its template position
    landmarks = place_face(600, 25, (900, 700))
    m = aligner.alignment_matrix(landmarks)
    mapped = landmarks @ m[:, :2].T + m[:, 2]
    assert abs(mapped[1, 1] - mapped[0, 1]) < 1e-6
    assert np.allclose(mapped, UTKFACE_TEMPLATE * 128, atol=1e-3)


def test_alignment_warp_puts_pixels_on_template(aligner):
    # 600 px face in a large photo: exercises the area-downsample path too
    image = np.full((1400, 1800, 3), 40, np.uint8)
    landmarks = place_face(600, -30, (900, 700))
    paint_eyes(image, landmarks, 25, (255, 0, 0))
    out = aligner.align(image, landmarks)
    assert out.shape == (128, 128, 3)
    for x, y in UTKFACE_TEMPLATE[:2] * 128:
        r, g, b = out[round(y), round(x)].astype(int)
        assert r > 200 and g < 60 and b < 60


def test_aligner_keeps_largest_face(aligner, monkeypatch):
    image = np.full((900, 1600, 3), 40, np.uint8)
    small, large = place_face(150, 0, (300, 300)), place_face(450, 10, (1100, 500))
    paint_eyes(image, small, 8, (255, 0, 0))
    paint_eyes(image, large, 20, (0, 255, 0))
    # Small face listed first, and more confident: size alone must decide
    rows = np.array([face_row(small, 150), face_row(large, 450)], dtype=np.float32)
    rows[0, 14] = 1.0
    monkeypatch.setattr(aligner, "detect_faces", lambda _: rows)
    out = np.asarray(aligner(Image.fromarray(image)))
    x, y = (UTKFACE_TEMPLATE[0] * 128).round().astype(int)
    r, g, b = out[y, x].astype(int)
    assert g > 200 and r < 60


def test_aligner_raises_when_no_face(aligner):
    flat = Image.new("RGB", (320, 240), (128, 100, 90))
    noise = Image.fromarray(np.random.default_rng(0).integers(0, 256, (240, 320, 3), dtype=np.uint8))
    for image in (flat, noise):
        with pytest.raises(NoFaceDetectedError):
            aligner(image)


def test_aligner_rejects_wrong_model_file(tmp_path):
    # e.g. the 131-byte git-LFS pointer saved in place of the model
    path = tmp_path / "face_detection_yunet_2026may.onnx"
    path.write_text("version https://git-lfs.github.com/spec/v1\n")
    with pytest.raises(RuntimeError, match="SHA-256"):
        FaceAligner(model_path=str(path))


def test_align_and_progress_returns_aligned_input_and_output(monkeypatch):
    progressor = AgeProgressor()
    landmarks = place_face(300, 15, (320, 240))
    rows = np.array([face_row(landmarks, 300)], dtype=np.float32)
    monkeypatch.setattr(progressor.aligner, "detect_faces", lambda _: rows)
    aligned, aged = progressor.align_and_progress(Image.new("RGB", (640, 480)), 3)
    assert aligned.size == aged.size == (128, 128)


@needs_utkface
def test_detect_faces_maps_downscaled_detection_back_to_photo(aligner):
    # A UTKFace crop enlarged 3x inside a phone-sized photo: detection runs
    # on a padded, downscaled copy, and must report original coordinates
    for name in sorted(os.listdir(UTKFACE_DIR))[:20]:
        face = cv2.imread(os.path.join(UTKFACE_DIR, name))
        ref = aligner.detect_faces(face)
        if len(ref):
            break
    photo = np.full((3000, 2400, 3), 90, np.uint8)
    photo[1000:1600, 700:1300] = cv2.resize(face, (600, 600), interpolation=cv2.INTER_CUBIC)
    found = aligner.detect_faces(photo)
    assert len(found) >= 1
    got = found[np.argmax(found[:, 2] * found[:, 3])]
    expected = ref[0, 4:14].reshape(5, 2) * 3 + [700, 1000]
    # Detector jitter differs between scales; a pad or scale bug is off by
    # hundreds of pixels
    assert np.abs(got[4:14].reshape(5, 2) - expected).max() < 30
