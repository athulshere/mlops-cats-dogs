"""Loading the checkpoint and turning raw image bytes into a prediction."""

import io

import torch
from PIL import Image, UnidentifiedImageError

from src.config import load_params, resolve
from src.dataset import build_transforms
from src.model import build_model, predict_probabilities, select_device


class InvalidImageError(ValueError):
    pass


class CatsDogsClassifier:
    def __init__(self, model_path=None, device=None):
        params = load_params()
        path = resolve(model_path or params["model"]["path"])
        if not path.exists():
            raise FileNotFoundError(f"No checkpoint at {path}. Train the model first.")

        checkpoint = torch.load(path, map_location="cpu")
        self.classes = checkpoint.get("classes", params["model"]["classes"])
        self.image_size = checkpoint.get("image_size", params["data"]["image_size"])
        self.device = device or select_device()

        self.model = build_model(num_classes=len(self.classes))
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device).eval()

        self.transform = build_transforms(self.image_size, augment=False)

    def to_tensor(self, image_bytes):
        try:
            image = Image.open(io.BytesIO(image_bytes))
            image.load()
        except (UnidentifiedImageError, OSError) as error:
            raise InvalidImageError("File could not be decoded as an image") from error

        if image.mode != "RGB":
            image = image.convert("RGB")
        return self.transform(image).unsqueeze(0)

    def predict(self, image_bytes):
        batch = self.to_tensor(image_bytes).to(self.device)
        probabilities = predict_probabilities(self.model, batch)[0].cpu()
        best = int(probabilities.argmax())

        return {
            "label": self.classes[best],
            "confidence": round(float(probabilities[best]), 4),
            "probabilities": {
                name: round(float(score), 4)
                for name, score in zip(self.classes, probabilities)
            },
        }
