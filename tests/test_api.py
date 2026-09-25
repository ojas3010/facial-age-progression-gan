"""Integration tests for the FastAPI backend: endpoint schema, validation
rules and error codes as specified in the project report (chapter 4)."""
import base64
import io
import os

import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from backend.main import NO_FACE_DETAIL, app, decode_upload
from ml_core.inference import UTKFACE_TEMPLATE

UTKFACE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "ml_core", "data", "utkface")


@pytest.fixture(scope="module")
def client():
    # Context manager triggers the lifespan hook, which initializes the model
    with TestClient(app) as c:
        yield c


def image_bytes(fmt="JPEG", size=(64, 64), mode="RGB", **save_kwargs):
    buf = io.BytesIO()
    Image.new(mode, size, color=0 if mode == "1" else (128, 100, 90)).save(buf, format=fmt, **save_kwargs)
    return buf.getvalue()


def jpeg_bytes(size=(64, 64)):
    return image_bytes("JPEG", size)


def post_image(client, data, content_type="image/jpeg", age="2"):
    return client.post(
        "/api/progress_age",
        files={"file": ("upload", data, content_type)},
        data={"target_age_group": age},
    )


def assert_rejected(r, detail_fragment):
    # Status alone is not enough: an upload with no face is also a 400, so a
    # broken validator would still "pass" via the face check
    assert r.status_code == 400
    assert detail_fragment in r.json()["detail"]


def decode_jpeg(b64):
    image = Image.open(io.BytesIO(base64.b64decode(b64)))
    assert image.format == "JPEG"
    return image


@pytest.fixture
def face_found(client, monkeypatch):
    """Stub the detector to report one face filling the image, so tests of
    the upload contract can use synthetic images. Only detection is faked;
    the alignment warp and the generator still run."""
    import backend.main as backend_main

    def detect_faces(image_bgr):
        side = min(image_bgr.shape[:2])
        landmarks = (UTKFACE_TEMPLATE * side).ravel()
        return np.array([[0, 0, side, side, *landmarks, 0.99]], dtype=np.float32)

    monkeypatch.setattr(backend_main.progressor.aligner, "detect_faces", detect_faces)


def test_health_check(client):
    import backend.main as backend_main

    r = client.get("/")
    assert r.status_code == 200
    # Lets the UI warn that output is noise when no checkpoint loaded
    assert r.json()["model_loaded"] is backend_main.progressor.weights_loaded


def test_progress_age_success(client, face_found):
    r = client.post(
        "/api/progress_age",
        files={"file": ("face.jpg", jpeg_bytes(), "image/jpeg")},
        data={"target_age_group": "2"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success"
    # Both payloads must decode to actual 128x128 JPEGs, not arbitrary bytes:
    # the aged face, and the aligned crop the UI shows beside it
    for key in ("image_base64", "aligned_image_base64"):
        assert isinstance(body[key], str) and len(body[key]) > 0
        assert decode_jpeg(body[key]).size == (128, 128)


def test_rejects_image_without_face(client):
    # Real detector: a flat colour and pure noise contain no face
    flat = image_bytes("JPEG", size=(320, 240))
    noise = io.BytesIO()
    Image.fromarray(np.random.default_rng(0).integers(0, 256, (240, 320, 3), dtype=np.uint8)).save(noise, "PNG")
    for data in (flat, noise.getvalue()):
        r = post_image(client, data)
        assert r.status_code == 400
        assert r.json()["detail"] == NO_FACE_DETAIL


@pytest.mark.skipif(not os.path.isdir(UTKFACE_DIR), reason="UTKFace not present (see README)")
def test_progress_age_real_face(client):
    # End to end with the real detector, no stubs
    name = sorted(f for f in os.listdir(UTKFACE_DIR) if f.endswith(".jpg"))[0]
    with open(os.path.join(UTKFACE_DIR, name), "rb") as f:
        r = post_image(client, f.read())
    assert r.status_code == 200
    body = r.json()
    assert decode_jpeg(body["image_base64"]).size == (128, 128)
    assert decode_jpeg(body["aligned_image_base64"]).size == (128, 128)


def test_rejects_unsupported_mime_type(client):
    r = client.post(
        "/api/progress_age",
        files={"file": ("evil.txt", b"not an image", "text/plain")},
        data={"target_age_group": "2"},
    )
    assert_rejected(r, "Unsupported file type")


def test_rejects_out_of_range_age_group(client):
    r = client.post(
        "/api/progress_age",
        files={"file": ("face.jpg", jpeg_bytes(), "image/jpeg")},
        data={"target_age_group": "9"},
    )
    assert r.status_code == 422


def test_rejects_non_integer_age_group(client):
    r = client.post(
        "/api/progress_age",
        files={"file": ("face.jpg", jpeg_bytes(), "image/jpeg")},
        data={"target_age_group": "old"},
    )
    assert r.status_code == 422


def test_rejects_oversized_payload(client):
    r = client.post(
        "/api/progress_age",
        files={"file": ("big.jpg", b"\xff" * (5 * 1024 * 1024 + 1), "image/jpeg")},
        data={"target_age_group": "2"},
    )
    assert_rejected(r, "File too large")


def test_rejects_corrupt_image_bytes(client):
    r = client.post(
        "/api/progress_age",
        files={"file": ("fake.jpg", b"definitely not jpeg data", "image/jpeg")},
        data={"target_age_group": "2"},
    )
    assert_rejected(r, "not a valid image")


def test_rejects_truncated_image(client):
    # Valid JPEG header, missing pixel data: fails at decode, not at open()
    data = jpeg_bytes()
    assert_rejected(post_image(client, data[: len(data) // 2]), "not a valid image")


def test_rejects_disallowed_format_behind_allowed_mime_type(client):
    # The client-declared content type is only a label; the decoder must
    # be restricted too
    assert_rejected(post_image(client, image_bytes("GIF"), "image/png"), "not a valid image")
    assert_rejected(post_image(client, image_bytes("TIFF"), "image/jpeg"), "not a valid image")


def test_accepts_png_labeled_as_jpeg(client, face_found):
    # Misnamed files are common; any allowed format passes under any allowed type
    assert post_image(client, image_bytes("PNG"), "image/jpeg").status_code == 200


def test_rejects_oversized_dimensions(client):
    # A few KB on the wire, 60 MP once decoded
    data = image_bytes("PNG", size=(10000, 6000), mode="1")
    assert len(data) < 1024 * 1024
    assert_rejected(post_image(client, data, "image/png"), "dimensions too large")


def test_decode_upload_applies_exif_orientation():
    # Orientation 6 = stored landscape, displayed portrait (typical phone photo)
    exif = Image.Exif()
    exif[0x0112] = 6
    data = image_bytes("JPEG", size=(60, 30), exif=exif.tobytes())
    image = decode_upload(data)
    assert image.size == (30, 60)
    assert image.mode == "RGB"


def test_returns_500_when_model_not_initialized(client, monkeypatch):
    import backend.main as backend_main

    monkeypatch.setattr(backend_main, "progressor", None)
    r = client.post(
        "/api/progress_age",
        files={"file": ("face.jpg", jpeg_bytes(), "image/jpeg")},
        data={"target_age_group": "2"},
    )
    assert r.status_code == 500
