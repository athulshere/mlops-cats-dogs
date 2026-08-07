import io

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from src.model import build_model


@pytest.fixture(scope="module")
def client(tmp_path_factory, monkeypatch_module=None):
    """Points the service at a throwaway checkpoint so the tests do not depend
    on whatever is currently sitting in models/."""
    import app.main as service

    path = tmp_path_factory.mktemp("api-model") / "model.pt"
    torch.save(
        {
            "state_dict": build_model(num_classes=2).state_dict(),
            "classes": ["cats", "dogs"],
            "image_size": 224,
            "architecture": "SimpleCNN",
        },
        path,
    )

    original = service.CatsDogsClassifier

    def factory(*args, **kwargs):
        kwargs.setdefault("model_path", path)
        return original(*args, **kwargs)

    service.CatsDogsClassifier = factory
    with TestClient(service.app) as test_client:
        yield test_client
    service.CatsDogsClassifier = original


def sample_upload():
    buffer = io.BytesIO()
    Image.new("RGB", (256, 256), (150, 90, 40)).save(buffer, format="JPEG")
    buffer.seek(0)
    return {"file": ("pet.jpg", buffer, "image/jpeg")}


def test_health_reports_the_model_is_loaded(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] is True


def test_predict_returns_label_and_probabilities(client):
    response = client.post("/predict", files=sample_upload())
    body = response.json()

    assert response.status_code == 200
    assert body["label"] in ("cats", "dogs")
    assert set(body["probabilities"]) == {"cats", "dogs"}


def test_predict_rejects_a_non_image_upload(client):
    response = client.post("/predict", files={"file": ("bad.txt", b"nope", "text/plain")})

    assert response.status_code == 400


def test_metrics_endpoint_counts_the_requests(client):
    client.post("/predict", files=sample_upload())
    response = client.get("/metrics")

    assert response.status_code == 200
    assert "inference_requests_total" in response.text
