"""Trains the baseline CNN and records the run in MLflow."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score
from torch import nn

from src.config import load_params, resolve
from src.dataset import build_loaders
from src.model import build_model, select_device


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)

    total_loss, correct, seen = 0.0, 0, 0
    predictions, targets = [], []

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            batch_predictions = outputs.argmax(dim=1)
            total_loss += loss.item() * labels.size(0)
            correct += (batch_predictions == labels).sum().item()
            seen += labels.size(0)
            predictions.extend(batch_predictions.cpu().tolist())
            targets.extend(labels.cpu().tolist())

    return {
        "loss": total_loss / max(seen, 1),
        "accuracy": correct / max(seen, 1),
        "predictions": predictions,
        "targets": targets,
    }


def plot_curves(history, out_path):
    epochs = range(1, len(history["train_loss"]) + 1)
    figure, (left, right) = plt.subplots(1, 2, figsize=(11, 4))

    left.plot(epochs, history["train_loss"], marker="o", label="train")
    left.plot(epochs, history["val_loss"], marker="o", label="validation")
    left.set_title("Loss")
    left.set_xlabel("epoch")
    left.legend()

    right.plot(epochs, history["train_accuracy"], marker="o", label="train")
    right.plot(epochs, history["val_accuracy"], marker="o", label="validation")
    right.set_title("Accuracy")
    right.set_xlabel("epoch")
    right.legend()

    figure.tight_layout()
    figure.savefig(out_path, dpi=120)
    plt.close(figure)


def plot_confusion(matrix, class_names, out_path):
    figure, axis = plt.subplots(figsize=(4.5, 4))
    axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(len(class_names)), class_names)
    axis.set_yticks(range(len(class_names)), class_names)
    axis.set_xlabel("predicted")
    axis.set_ylabel("actual")
    axis.set_title("Confusion matrix (test set)")

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, int(matrix[row, column]), ha="center", va="center")

    figure.tight_layout()
    figure.savefig(out_path, dpi=120)
    plt.close(figure)


def main():
    params = load_params()
    data_params, train_params, model_params, mlflow_params = (
        params["data"],
        params["train"],
        params["model"],
        params["mlflow"],
    )

    parser = argparse.ArgumentParser(description="Train the cats vs dogs baseline")
    parser.add_argument("--epochs", type=int, default=train_params["epochs"])
    parser.add_argument("--batch-size", type=int, default=train_params["batch_size"])
    parser.add_argument("--learning-rate", type=float, default=train_params["learning_rate"])
    parser.add_argument("--run-name", default="baseline-cnn")
    args = parser.parse_args()

    torch.manual_seed(train_params["seed"])
    device = select_device()
    print(f"Training on {device}")

    processed_dir = resolve(data_params["processed_dir"])
    if not processed_dir.exists():
        raise SystemExit("Processed data missing - run `python -m src.data_prep` first.")

    loaders, class_names = build_loaders(
        processed_dir,
        image_size=data_params["image_size"],
        batch_size=args.batch_size,
        num_workers=train_params["num_workers"],
    )

    model = build_model(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=train_params["weight_decay"],
    )

    reports_dir = resolve("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    model_path = resolve(model_params["path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(mlflow_params["tracking_uri"])
    mlflow.set_experiment(mlflow_params["experiment_name"])

    history = {key: [] for key in ("train_loss", "val_loss", "train_accuracy", "val_accuracy")}
    best_val_accuracy = -1.0

    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_params(
            {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "weight_decay": train_params["weight_decay"],
                "image_size": data_params["image_size"],
                "architecture": "SimpleCNN-4blocks",
                "optimizer": "Adam",
                "augmentation": "flip+rotation+color-jitter",
                "train_images": len(loaders["train"].dataset),
                "val_images": len(loaders["val"].dataset),
                "test_images": len(loaders["test"].dataset),
            }
        )

        for epoch in range(1, args.epochs + 1):
            train_stats = run_epoch(model, loaders["train"], criterion, device, optimizer)
            val_stats = run_epoch(model, loaders["val"], criterion, device)

            history["train_loss"].append(train_stats["loss"])
            history["val_loss"].append(val_stats["loss"])
            history["train_accuracy"].append(train_stats["accuracy"])
            history["val_accuracy"].append(val_stats["accuracy"])

            mlflow.log_metrics(
                {
                    "train_loss": train_stats["loss"],
                    "train_accuracy": train_stats["accuracy"],
                    "val_loss": val_stats["loss"],
                    "val_accuracy": val_stats["accuracy"],
                },
                step=epoch,
            )
            print(
                f"epoch {epoch}/{args.epochs} "
                f"train_loss={train_stats['loss']:.4f} train_acc={train_stats['accuracy']:.4f} "
                f"val_loss={val_stats['loss']:.4f} val_acc={val_stats['accuracy']:.4f}"
            )

            if val_stats["accuracy"] > best_val_accuracy:
                best_val_accuracy = val_stats["accuracy"]
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "classes": class_names,
                        "image_size": data_params["image_size"],
                        "architecture": "SimpleCNN",
                    },
                    model_path,
                )

        # Score the checkpoint we actually kept, not the last epoch.
        model.load_state_dict(torch.load(model_path, map_location=device)["state_dict"])
        test_stats = run_epoch(model, loaders["test"], criterion, device)
        test_f1 = f1_score(test_stats["targets"], test_stats["predictions"], average="macro")

        matrix = confusion_matrix(test_stats["targets"], test_stats["predictions"])
        curves_path = reports_dir / "loss_curves.png"
        confusion_path = reports_dir / "confusion_matrix.png"
        plot_curves(history, curves_path)
        plot_confusion(np.array(matrix), class_names, confusion_path)

        metrics = {
            "best_val_accuracy": best_val_accuracy,
            "test_accuracy": test_stats["accuracy"],
            "test_loss": test_stats["loss"],
            "test_f1_macro": test_f1,
        }
        (reports_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(curves_path))
        mlflow.log_artifact(str(confusion_path))
        mlflow.log_artifact(str(reports_dir / "metrics.json"))
        mlflow.log_artifact(str(model_path), artifact_path="model")

        print(f"test accuracy={test_stats['accuracy']:.4f} f1={test_f1:.4f}")
        print(f"checkpoint saved to {model_path}")


if __name__ == "__main__":
    main()
