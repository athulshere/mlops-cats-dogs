"""Post-deploy check: is the service up, and does it actually predict?

Exits non-zero on any failure so the CD pipeline stops there.
"""

import argparse
import io
import sys
import time
from pathlib import Path

import requests
from PIL import Image


def wait_for_health(base_url, timeout):
    deadline = time.time() + timeout
    last_error = "no response"

    while time.time() < deadline:
        try:
            response = requests.get(f"{base_url}/health", timeout=5)
            if response.status_code == 200 and response.json().get("model_loaded"):
                print(f"health ok: {response.json()}")
                return True
            last_error = f"status={response.status_code} body={response.text[:120]}"
        except requests.RequestException as error:
            last_error = str(error)
        time.sleep(3)

    print(f"health check never passed within {timeout}s ({last_error})")
    return False


def sample_image(path=None):
    if path and Path(path).exists():
        return Path(path).read_bytes()

    buffer = io.BytesIO()
    Image.new("RGB", (224, 224), (170, 120, 80)).save(buffer, format="JPEG")
    return buffer.getvalue()


def check_prediction(base_url, image_path=None):
    files = {"file": ("smoke.jpg", sample_image(image_path), "image/jpeg")}
    response = requests.post(f"{base_url}/predict", files=files, timeout=30)

    if response.status_code != 200:
        print(f"prediction failed: status={response.status_code} body={response.text[:200]}")
        return False

    body = response.json()
    if body.get("label") not in ("cats", "dogs"):
        print(f"unexpected label in response: {body}")
        return False
    if not body.get("probabilities"):
        print(f"response carried no probabilities: {body}")
        return False

    print(f"prediction ok: {body}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Smoke test the deployed service")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--image", help="Optional real image to send instead of a blank one")
    args = parser.parse_args()

    base_url = args.url.rstrip("/")
    print(f"smoke testing {base_url}")

    if not wait_for_health(base_url, args.timeout):
        sys.exit(1)
    if not check_prediction(base_url, args.image):
        sys.exit(1)

    print("smoke tests passed")


if __name__ == "__main__":
    main()
