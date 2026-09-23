"""Integration tests for the FastAPI backend: endpoint schema, validation
rules and error codes as specified in the project report (chapter 4)."""
import base64
import io

import pytest
from PIL import Image
from fastapi.testclient import TestClient

from backend.main import app, decode_upload


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


def test_health_check(client):
    import backend.main as backend_main

    r = client.get("/")
    assert r.status_code == 200
    # Lets the UI warn that output is noise when no checkpoint loaded
    assert r.json()["model_loaded"] is backend_main.progressor.weights_loaded


def test_progress_age_success(client):
    r = client.post(
        "/api/progress_age",
        files={"file": ("face.jpg", jpeg_bytes(), "image/jpeg")},
        data={"target_age_group": "2"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success"
    assert isinstance(body["image_base64"], str) and len(body["image_base64"]) > 0
    # Payload must decode to an actual 128x128 JPEG, not arbitrary bytes
    out = Image.open(io.BytesIO(base64.b64decode(body["image_base64"])))
    assert out.format == "JPEG"
    assert out.size == (128, 128)


def test_rejects_unsupported_mime_type(client):
    r = client.post(
        "/api/progress_age",
        files={"file": ("evil.txt", b"not an image", "text/plain")},
        data={"target_age_group": "2"},
    )
    assert r.status_code == 400


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
    assert r.status_code == 400


def test_rejects_corrupt_image_bytes(client):
    r = client.post(
        "/api/progress_age",
        files={"file": ("fake.jpg", b"definitely not jpeg data", "image/jpeg")},
        data={"target_age_group": "2"},
    )
    assert r.status_code == 400


def test_rejects_truncated_image(client):
    # Valid JPEG header, missing pixel data: fails at decode, not at open()
    data = jpeg_bytes()
    assert post_image(client, data[: len(data) // 2]).status_code == 400


def test_rejects_disallowed_format_behind_allowed_mime_type(client):
    # The client-declared content type is only a label; the decoder must
    # be restricted too
    assert post_image(client, image_bytes("GIF"), "image/png").status_code == 400
    assert post_image(client, image_bytes("TIFF"), "image/jpeg").status_code == 400


def test_accepts_png_labeled_as_jpeg(client):
    # Misnamed files are common; any allowed format passes under any allowed type
    assert post_image(client, image_bytes("PNG"), "image/jpeg").status_code == 200


def test_rejects_oversized_dimensions(client):
    # A few KB on the wire, 60 MP once decoded
    data = image_bytes("PNG", size=(10000, 6000), mode="1")
    assert len(data) < 1024 * 1024
    assert post_image(client, data, "image/png").status_code == 400


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
