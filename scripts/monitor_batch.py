"""Replays a batch of labelled test images through the deployed service and
compares the live predictions against the true labels.

This is the post-deployment performance check - it tells us whether the model
behind the API is still doing what it did at training time.
"""

import argparse
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

CLASSES = ("cats", "dogs")


def gather_samples(test_dir, per_class, seed=42):
    rng = random.Random(seed)
    samples = []

    for label in CLASSES:
        folder = Path(test_dir) / label
        files = sorted(folder.glob("*.jpg"))
        if not files:
            raise SystemExit(f"No images under {folder} - run the data prep stage first.")
        rng.shuffle(files)
        samples.extend((path, label) for path in files[:per_class])

    rng.shuffle(samples)
    return samples


def call_service(base_url, image_path):
    with open(image_path, "rb") as handle:
        files = {"file": (image_path.name, handle, "image/jpeg")}
        started = time.perf_counter()
        response = requests.post(f"{base_url}/predict", files=files, timeout=30)
        latency_ms = (time.perf_counter() - started) * 1000

    response.raise_for_status()
    return response.json(), latency_ms


def main():
    parser = argparse.ArgumentParser(description="Post-deployment performance check")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--test-dir", default="data/processed/test")
    parser.add_argument("--per-class", type=int, default=25)
    parser.add_argument("--out", default="reports/post_deploy_report.json")
    args = parser.parse_args()

    base_url = args.url.rstrip("/")
    samples = gather_samples(args.test_dir, args.per_class)

    correct = 0
    latencies = []
    confusion = {actual: {predicted: 0 for predicted in CLASSES} for actual in CLASSES}
    rows = []

    for image_path, true_label in samples:
        body, latency_ms = call_service(base_url, image_path)
        predicted = body["label"]

        confusion[true_label][predicted] += 1
        latencies.append(latency_ms)
        correct += int(predicted == true_label)
        rows.append(
            {
                "file": image_path.name,
                "true_label": true_label,
                "predicted": predicted,
                "confidence": body["confidence"],
                "latency_ms": round(latency_ms, 2),
            }
        )

    latencies.sort()
    total = len(samples)
    report = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "endpoint": base_url,
        "requests": total,
        "accuracy": round(correct / total, 4),
        "latency_ms": {
            "mean": round(sum(latencies) / total, 2),
            "p50": round(latencies[total // 2], 2),
            "p95": round(latencies[min(int(total * 0.95), total - 1)], 2),
            "max": round(latencies[-1], 2),
        },
        "confusion_matrix": confusion,
        "samples": rows,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))

    print(f"{total} requests, live accuracy {report['accuracy']:.4f}")
    print(f"latency mean={report['latency_ms']['mean']}ms p95={report['latency_ms']['p95']}ms")
    print(f"report written to {out_path}")


if __name__ == "__main__":
    main()
