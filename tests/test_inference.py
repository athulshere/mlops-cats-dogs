import io

import pytest
import torch
from PIL import Image

from src.inference import CatsDogsClassifier, InvalidImageError
from src.model import build_model, predict_probabilities


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    """An untrained checkpoint is enough to exercise the loading and predict path."""
    path = tmp_path_factory.mktemp("models") / "test_model.pt"
    model = build_model(num_classes=2)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "classes": ["cats", "dogs"],
            "image_size": 224,
            "architecture": "SimpleCNN",
        },
        path,
    )
    return path


@pytest.fixture(scope="module")
def classifier(checkpoint):
    return CatsDogsClassifier(model_path=checkpoint, device=torch.device("cpu"))


def image_bytes(size=(300, 200), colour=(90, 140, 200)):
    buffer = io.BytesIO()
    Image.new("RGB", size, colour).save(buffer, format="JPEG")
    return buffer.getvalue()


def test_predict_probabilities_returns_a_valid_distribution():
    model = build_model(num_classes=2)
    batch = torch.rand(4, 3, 224, 224)

    probabilities = predict_probabilities(model, batch)

    assert probabilities.shape == (4, 2)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(4), atol=1e-5)
    assert (probabilities >= 0).all()


def test_predict_probabilities_does_not_track_gradients():
    model = build_model(num_classes=2)

    probabilities = predict_probabilities(model, torch.rand(1, 3, 224, 224))

    assert not probabilities.requires_grad


def test_to_tensor_normalises_shape_for_any_input_size(classifier):
    tensor = classifier.to_tensor(image_bytes(size=(640, 111)))

    assert tensor.shape == (1, 3, 224, 224)


def test_predict_returns_a_known_label_with_probabilities(classifier):
    result = classifier.predict(image_bytes())

    assert result["label"] in ("cats", "dogs")
    assert set(result["probabilities"]) == {"cats", "dogs"}
    assert 0.0 <= result["confidence"] <= 1.0
    assert round(sum(result["probabilities"].values()), 2) == 1.0


def test_predict_rejects_something_that_is_not_an_image(classifier):
    with pytest.raises(InvalidImageError):
        classifier.predict(b"this is not a jpeg")


def test_missing_checkpoint_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError):
        CatsDogsClassifier(model_path=tmp_path / "nope.pt")
