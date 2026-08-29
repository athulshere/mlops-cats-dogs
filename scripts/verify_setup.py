"""Checks the local machine has everything the pipeline needs.

Run this first - it is faster to find a missing Docker daemon here than
halfway through a build.
"""

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
results = []


def record(name, ok, detail):
    results.append((name, ok, detail))


def check_python():
    version = sys.version_info
    ok = version >= (3, 10)
    record("Python 3.10+", ok, f"{version.major}.{version.minor}.{version.micro}")


def check_imports():
    for package in ("torch", "torchvision", "fastapi", "mlflow", "pytest", "yaml"):
        try:
            module = __import__(package)
            version = getattr(module, "__version__", "installed")
            record(f"import {package}", True, version)
        except ImportError:
            record(f"import {package}", False, "not installed")


def check_command(name, args):
    if shutil.which(args[0]) is None:
        record(name, False, "not on PATH")
        return
    try:
        output = subprocess.run(args, capture_output=True, text=True, timeout=25)
        first_line = (output.stdout or output.stderr).strip().splitlines()[0]
        record(name, output.returncode == 0, first_line[:70])
    except (subprocess.TimeoutExpired, OSError) as error:
        record(name, False, str(error)[:70])


def check_dataset():
    raw = ROOT / "data" / "raw"
    count = sum(1 for _ in raw.rglob("*.jpg")) if raw.exists() else 0
    record("Dataset in data/raw", count > 100, f"{count} jpg files")


def check_model():
    path = ROOT / "models" / "cats_dogs_cnn.pt"
    if path.exists():
        record("Trained checkpoint", True, f"{path.stat().st_size / 1e6:.1f} MB")
    else:
        record("Trained checkpoint", False, "missing - run training")


def main():
    check_python()
    check_imports()
    check_command("Docker daemon", ["docker", "info", "--format", "{{.ServerVersion}}"])
    check_command("Docker Compose", ["docker", "compose", "version"])
    check_command("Git", ["git", "--version"])
    check_command("DVC", ["dvc", "--version"])
    check_dataset()
    check_model()

    print()
    for name, ok, detail in results:
        print(f"  {'[ok]  ' if ok else '[FAIL]'} {name:22s} {detail}")

    blockers = [name for name, ok, _ in results if not ok]
    print()
    if blockers:
        print("Not ready yet: " + ", ".join(blockers))
        print("The dataset and checkpoint are expected to fail before you train.")
    else:
        print("Everything checks out.")


if __name__ == "__main__":
    main()
