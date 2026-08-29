"""Checks the repo is consistent and ready to push.

Catches the things that only show up as a red X ten minutes into a CI run -
mismatched ports, a checkpoint that never got committed, the placeholder still
sitting in the Kubernetes manifest.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
checks = []


def add(name, ok, detail, blocking=True):
    checks.append((name, ok, detail, blocking))


def git(*args):
    result = subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def read(relative):
    path = ROOT / relative
    return path.read_text() if path.exists() else ""


def check_branch():
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    add("On branch main", branch == "main", branch or "not a git repo")


def check_clean_tree():
    status = git("status", "--porcelain")
    if status is None:
        add("Working tree committed", False, "not a git repo")
        return
    pending = [line for line in status.splitlines() if line.strip()]
    add(
        "Working tree committed",
        not pending,
        "clean" if not pending else f"{len(pending)} uncommitted file(s)",
    )


def check_remote():
    remote = git("remote", "get-url", "origin")
    add("Remote origin set", bool(remote), remote or "no origin yet")


def check_model_tracked():
    tracked = git("ls-files", "models/") or ""
    committed = "cats_dogs_cnn.pt" in tracked
    on_disk = (ROOT / "models" / "cats_dogs_cnn.pt").exists()

    if committed:
        detail = "committed"
    elif on_disk:
        detail = "on disk but NOT committed - run: git add -f models/cats_dogs_cnn.pt"
    else:
        detail = "missing - train the model first"
    add("Checkpoint in git", committed, detail)


def compose_host_port():
    match = re.search(r'-\s*"(\d+):(\d+)"', read("docker-compose.yml"))
    return match.groups() if match else (None, None)


def workflow_port():
    match = re.search(r"smoke_test\.py --url http://localhost:(\d+)", read(".github/workflows/ci-cd.yml"))
    return match.group(1) if match else None


def check_ports():
    host, container = compose_host_port()
    ci = workflow_port()

    if host is None:
        add("Compose port mapping", False, "no ports line found")
        return

    add("Container port is 8000", container == "8000", f"compose maps {host} -> {container}")
    add(
        "CI smoke test port matches compose",
        ci == host,
        f"compose publishes {host}, workflow tests {ci}",
    )


def check_k8s_image():
    text = read("k8s/deployment.yaml")
    match = re.search(r"image:\s*(\S+)", text)
    image = (match.group(1) if match else "").strip()

    placeholder = "OWNER" in image or "REPO" in image
    add(
        "k8s image path filled in",
        bool(image) and not placeholder,
        image or "no image line",
        blocking=False,
    )

    # A registry reference is only letters, digits and . _ - / : @
    stray = set(re.findall(r"[^A-Za-z0-9._\-/:@]", image))
    if image and stray:
        add("k8s image has no stray characters", False, f"{image} contains {''.join(sorted(stray))}")
    if image and image != image.lower():
        add("k8s image is lowercase", False, image, blocking=False)


def check_pins():
    unpinned = []
    for name in ("requirements.txt", "requirements-dev.txt"):
        for line in read(name).splitlines():
            line = line.strip()
            if not line or line.startswith(("#", "-r")):
                continue
            if "==" not in line:
                unpinned.append(f"{name}: {line}")
    add("All dependencies pinned", not unpinned, "; ".join(unpinned) or "every line uses ==")


def check_dockerfile():
    text = read("Dockerfile")
    add("Dockerfile copies models/", "COPY models/" in text, "found" if "COPY models/" in text else "missing")


def check_artifacts():
    metrics = ROOT / "reports" / "metrics.json"
    if metrics.exists():
        data = json.loads(metrics.read_text())
        accuracy = data.get("test_accuracy", 0)
        add("Training metrics present", True, f"test accuracy {accuracy:.3f}")
        if accuracy < 0.75:
            add("Accuracy above 0.75", False, f"{accuracy:.3f} - consider more epochs", blocking=False)
    else:
        add("Training metrics present", False, "reports/metrics.json missing")

    report = ROOT / "reports" / "post_deploy_report.json"
    if report.exists():
        data = json.loads(report.read_text())
        add(
            "Post-deploy report present",
            True,
            f"{data.get('requests')} requests, live accuracy {data.get('accuracy')}",
        )
    else:
        add("Post-deploy report present", False, "run scripts/monitor_batch.py")

    for name in ("confusion_matrix.png", "loss_curves.png"):
        add(f"reports/{name}", (ROOT / "reports" / name).exists(), "", blocking=False)


def check_module_files():
    required = {
        "M1 tracking": "src/train.py",
        "M1 versioning": "dvc.yaml",
        "M2 service": "app/main.py",
        "M2 container": "Dockerfile",
        "M3 tests": "tests/test_data_prep.py",
        "M3 pipeline": ".github/workflows/ci-cd.yml",
        "M4 manifests": "k8s/deployment.yaml",
        "M4 smoke test": "scripts/smoke_test.py",
        "M5 monitoring": "scripts/monitor_batch.py",
    }
    missing = [f"{label} ({path})" for label, path in required.items() if not (ROOT / path).exists()]
    add("All module files present", not missing, "; ".join(missing) or f"{len(required)} files")


def main():
    check_branch()
    check_clean_tree()
    check_remote()
    check_model_tracked()
    check_ports()
    check_k8s_image()
    check_pins()
    check_dockerfile()
    check_artifacts()
    check_module_files()

    print()
    for name, ok, detail, blocking in checks:
        mark = "[ok]  " if ok else ("[FAIL]" if blocking else "[warn]")
        print(f"  {mark} {name:34s} {detail}")

    blockers = [name for name, ok, _, blocking in checks if not ok and blocking]
    warnings = [name for name, ok, _, blocking in checks if not ok and not blocking]

    print()
    if blockers:
        print(f"{len(blockers)} problem(s) to fix before pushing:")
        for name in blockers:
            print(f"  - {name}")
        sys.exit(1)

    if warnings:
        print(f"Ready to push, with {len(warnings)} thing(s) worth a look: {', '.join(warnings)}")
    else:
        print("Everything checks out. Push it.")


if __name__ == "__main__":
    main()
