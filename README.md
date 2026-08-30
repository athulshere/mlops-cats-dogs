# Cats vs Dogs — End-to-End MLOps Pipeline

Binary image classifier for a pet adoption platform, wired up with experiment
tracking, a containerised inference API, CI/CD on GitHub Actions, and basic
post-deployment monitoring.

**Course:** MLOps (S1-25_AIMLCZG523) · Assignment 2

Repository: https://github.com/athulshere/mlops-cats-dogs

Screen recording: recording/MLOPS_Assignment_Recording.mp4 (4 min 43 s) .
Also on Drive: https://drive.google.com/file/d/1P2dymokxOfR4sF242FEmqWRdgIO21PZa/view?usp=drive_link

---

## What's in here

```
├── src/                 data prep, model, training, inference
├── app/main.py          FastAPI inference service
├── tests/               pytest suite (16 tests)
├── scripts/             smoke test + post-deployment monitoring
├── k8s/                 Deployment + Service manifests
├── .github/workflows/   CI/CD pipeline
├── dvc.yaml             prepare -> train pipeline
├── params.yaml          all tunables in one place
├── Dockerfile           CPU-only inference image
└── docker-compose.yml   deployment target
```

---

## Setup

Step-by-step local instructions with troubleshooting are in **[RUNBOOK.md](RUNBOOK.md)**.

```bash
python -m venv .venv && source .venv/bin/activate
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-dev.txt
```

Download the [Kaggle Cats vs Dogs dataset](https://www.kaggle.com/datasets/salader/dogs-vs-cats)
and unzip it into `data/raw/`. Any of the common layouts works — the loader looks
at both folder names and file names, so `PetImages/Cat/*.jpg` and
`train/cat.0.jpg` are both fine.

---

## M1 — Model development and experiment tracking

Version the raw data and run the pipeline:

```bash
dvc add data/raw
git add data/raw.dvc data/.gitignore && git commit -m "Track raw dataset with DVC"
dvc push

dvc repro                 # runs prepare, then train
```

Or run the two stages directly:

```bash
python -m src.data_prep   # 224x224 RGB, 80/10/10 split, class-balanced
python -m src.train       # trains, logs to MLflow, writes models/cats_dogs_cnn.pt
```

`params.yaml` caps the dataset at 3000 images per class so a CPU/MPS run finishes
in a sensible time. Set `limit_per_class: 0` to train on everything.

Inspect the runs:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Logged per run: hyperparameters, dataset sizes, per-epoch train/val loss and
accuracy, final test accuracy and macro-F1, plus the confusion matrix, loss
curves, `metrics.json` and the model checkpoint as artifacts.

> The Docker build copies `models/`, so commit `models/cats_dogs_cnn.pt` (~1 MB)
> after training — CI needs it to build the image.

---

## M2 — Packaging and containerisation

Run the API locally:

```bash
uvicorn app.main:app --reload --port 8000
```

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Liveness/readiness, reports whether the model loaded |
| `/predict` | POST | Accepts an image upload, returns label + class probabilities |
| `/stats` | GET | Request count, error count, average latency, per-class counts |
| `/metrics` | GET | The same counters in Prometheus text format |
| `/docs` | GET | Swagger UI |

Build and run the container:

```bash
docker build -t cats-dogs-api:local .
docker compose up -d

curl http://localhost:8000/health
curl -X POST -F "file=@$(ls data/processed/test/dogs/*.jpg | head -1)" http://localhost:8000/predict
```

Sample response:

```json
{
  "label": "dogs",
  "confidence": 0.8932,
  "probabilities": {"cats": 0.1068, "dogs": 0.8932},
  "inference_ms": 66.47
}
```

Every key library is pinned in `requirements.txt` (verified on Python 3.11-3.13). The Dockerfile installs torch
from the CPU wheel index — the default Linux wheel drags in the whole CUDA stack
and adds several gigabytes for no benefit.

---

## M3 — CI pipeline

```bash
pytest -v
```

Sixteen tests across three files:

- `tests/test_data_prep.py` — label resolution, image collection, the 224×224 RGB
  preprocessing step, split ratios, split reproducibility, no train/test leakage
- `tests/test_inference.py` — softmax output shape and normalisation, no gradient
  leakage, tensor shaping for arbitrary input sizes, prediction payload, error
  handling for corrupt uploads and missing checkpoints
- `tests/test_api.py` — health, predict, bad upload rejection, metrics endpoint

`.github/workflows/ci-cd.yml` runs on every push and pull request to `main`:

1. **test** — checkout, Python 3.11, install pinned deps, run pytest
2. **build-and-push** — Buildx build, tagged `latest` and `sha-<short>`, pushed to
   GitHub Container Registry (`ghcr.io/<owner>/<repo>`) with layer caching

Pull requests build the image but don't publish it.

---

## M4 — CD pipeline and deployment

The **deploy** job runs only on `main`, after a successful build:

1. Pull the newly published image from GHCR
2. `docker compose up -d` to roll the service over
3. `python scripts/smoke_test.py` — waits for `/health` to report a loaded model,
   then sends a real prediction request and validates the response shape
4. Dump container logs if anything failed, then tear down

A non-zero exit from the smoke test fails the job, so a broken image never gets
marked as deployed.

Kubernetes is available as an alternative target:

```bash
kind create cluster --name mlops
kind load docker-image cats-dogs-api:local --name mlops
kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml
kubectl rollout status deployment/cats-dogs-api
```

Two replicas, readiness and liveness probes on `/health`, NodePort 30080. Update
the image reference in `k8s/deployment.yaml` to your GHCR path before applying.

---

## M5 — Monitoring and post-deployment tracking

The service logs one JSON line per request to `logs/requests.log` — timestamp,
request id, method, path, status, latency. Predictions add the filename, label
and confidence. **Image bytes are never written to the log.**

```json
{"timestamp": "2026-08-07T18:35:10Z", "request_id": "167c5bb5", "method": "POST", "path": "/predict", "status": 200, "latency_ms": 52.97}
{"event": "prediction", "filename": "dogs_00001.jpg", "label": "dogs", "confidence": 0.8932, "inference_ms": 66.47}
```

In-app counters are exposed on `/stats` (JSON) and `/metrics` (Prometheus text),
so a Prometheus scrape config can be pointed at the service without extra code.

Post-deployment performance check — replays a labelled batch from the held-out
test set through the live endpoint and compares predictions to true labels:

```bash
python scripts/monitor_batch.py --url http://localhost:8000 --per-class 25
```

Writes `reports/post_deploy_report.json` with live accuracy, a confusion matrix,
and mean/p50/p95/max latency. Comparing that accuracy against the training-time
figure in `reports/metrics.json` is the drift signal.

---

## Demo walkthrough (for the screen recording)

1. `mlflow ui --backend-store-uri sqlite:///mlflow.db` — show runs, metrics,
   confusion matrix and loss curves
2. `git log --oneline` and `dvc.yaml` — code and data versioning
3. Make a small code change (bump `APP_VERSION` in `app/main.py`), commit, push
4. GitHub Actions — tests pass, image builds, image appears in GHCR Packages
5. Deploy job pulls the image, brings the service up, smoke test passes
6. `curl` the health and predict endpoints against the running container
7. `curl /metrics` and `tail logs/requests.log` — monitoring
8. `python scripts/monitor_batch.py` — live accuracy vs training accuracy

---

## Handy commands

```bash
make verify         # check the machine has what it needs
make install        # dependencies
make prepare        # data prep
make train          # training + MLflow
make test           # pytest
make docker-build   # build image
make docker-run     # docker compose up
make smoke          # smoke test
make monitor        # post-deployment check
```
