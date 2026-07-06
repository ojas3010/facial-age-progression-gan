"""Integration tests for the FastAPI backend: endpoint schema, validation
rules and error codes as specified in the project report (chapter 4)."""
import io

import pytest
from PIL import Image
from fastapi.testclient import TestClient

from backend.main import app


@pytest.fixture(scope="module")
def client():
    # Context manager triggers the lifespan hook, which initializes the model
    with TestClient(app) as c:
        yield c


def jpeg_bytes(size=(64, 64)):
    buf = io.BytesIO()
    Image.new("RGB", size, color=(128, 100, 90)).save(buf, format="JPEG")
    return buf.getvalue()


def test_health_check(client):
    r = client.get("/")
    assert r.status_code == 200


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
