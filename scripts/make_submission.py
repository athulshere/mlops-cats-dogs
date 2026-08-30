"""Builds the submission zip and checks what actually landed inside it.

Excludes the image folders and the virtual environment, keeps the trained
model, the reports and the MLflow database.
"""

import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT.parent / f"{ROOT.name}-submission.zip"

EXCLUDE = [
    "*/.venv/*",
    "*/.git/*",
    "*/data/raw/*",
    "*/data/processed/*",
    "*/__pycache__/*",
    "*/.pytest_cache/*",
    "*/mlartifacts/*",
    ".DS_Store",
    "*/.DS_Store",
]

MUST_CONTAIN = [
    ("Trained model", "models/cats_dogs_cnn.pt"),
    ("Training metrics", "reports/metrics.json"),
    ("Confusion matrix", "reports/confusion_matrix.png"),
    ("Loss curves", "reports/loss_curves.png"),
    ("Post-deploy report", "reports/post_deploy_report.json"),
    ("MLflow database", "mlflow.db"),
    ("Inference service", "app/main.py"),
    ("Training script", "src/train.py"),
    ("Dockerfile", "Dockerfile"),
    ("Compose file", "docker-compose.yml"),
    ("CI/CD workflow", ".github/workflows/ci-cd.yml"),
    ("DVC pipeline", "dvc.yaml"),
    ("k8s deployment", "k8s/deployment.yaml"),
    ("k8s service", "k8s/service.yaml"),
    ("Requirements", "requirements.txt"),
    ("Tests", "tests/test_data_prep.py"),
    ("Smoke test", "scripts/smoke_test.py"),
    ("Monitoring script", "scripts/monitor_batch.py"),
    ("README", "README.md"),
]


def build():
    if ARCHIVE.exists():
        ARCHIVE.unlink()

    command = ["zip", "-qr", str(ARCHIVE), ROOT.name, "-x", *EXCLUDE]
    result = subprocess.run(command, cwd=ROOT.parent, check=False)
    if result.returncode != 0:
        print("zip failed")
        sys.exit(1)


def verify():
    with zipfile.ZipFile(ARCHIVE) as archive:
        names = set(archive.namelist())
        size_mb = sum(item.file_size for item in archive.infolist()) / 1e6
        count = len(names)

    missing = []
    print()
    for label, relative in MUST_CONTAIN:
        entry = f"{ROOT.name}/{relative}"
        present = entry in names
        print(f"  {'[ok]  ' if present else '[FAIL]'} {label:22s} {relative}")
        if not present:
            missing.append(relative)

    print(f"\n  {count} files, {size_mb:.1f} MB uncompressed")
    print(f"  {ARCHIVE}")

    leaked = [name for name in names if "/data/raw/" in name or "/.venv/" in name]
    if leaked:
        print(f"\n  {len(leaked)} file(s) that should have been excluded slipped in")

    if missing:
        print(f"\n{len(missing)} thing(s) missing from the archive:")
        for item in missing:
            print(f"  - {item}")
        sys.exit(1)

    print("\nArchive is complete.")


if __name__ == "__main__":
    build()
    verify()
