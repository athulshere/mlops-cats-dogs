"""REST wrapper around the trained classifier."""

import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import PlainTextResponse

from src.inference import CatsDogsClassifier, InvalidImageError

APP_VERSION = os.getenv("APP_VERSION", "1.0.0")
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("cats-dogs-api")
logger.setLevel(logging.INFO)
if not logger.handlers:
    file_handler = logging.FileHandler(LOG_DIR / "requests.log")
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(file_handler)
    logger.addHandler(logging.StreamHandler())


class Metrics:
    """In-app counters, exposed in Prometheus text format on /metrics."""

    def __init__(self):
        self._lock = Lock()
        self.requests_total = 0
        self.errors_total = 0
        self.predictions = {}
        self.latency_sum_seconds = 0.0

    def record(self, latency_seconds, label=None, failed=False):
        with self._lock:
            self.requests_total += 1
            self.latency_sum_seconds += latency_seconds
            if failed:
                self.errors_total += 1
            if label:
                self.predictions[label] = self.predictions.get(label, 0) + 1

    def snapshot(self):
        with self._lock:
            average = self.latency_sum_seconds / self.requests_total if self.requests_total else 0.0
            return {
                "requests_total": self.requests_total,
                "errors_total": self.errors_total,
                "predictions": dict(self.predictions),
                "average_latency_ms": round(average * 1000, 2),
            }


metrics = Metrics()
state = {"classifier": None}


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Load once at boot so the first request isn't paying for it.
    state["classifier"] = CatsDogsClassifier()
    logger.info(json.dumps({"event": "startup", "classes": state["classifier"].classes}))
    yield
    state["classifier"] = None


app = FastAPI(
    title="Cats vs Dogs Classifier",
    description="Baseline CNN served for the pet adoption platform",
    version=APP_VERSION,
    lifespan=lifespan,
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    started = time.perf_counter()
    request_id = str(uuid.uuid4())[:8]
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - started) * 1000

    # Only request metadata is logged - the uploaded image itself never lands in the log.
    logger.info(
        json.dumps(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "latency_ms": round(elapsed_ms, 2),
            }
        )
    )
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time-ms"] = f"{elapsed_ms:.2f}"
    return response


@app.get("/health")
def health():
    ready = state["classifier"] is not None
    return {
        "status": "ok" if ready else "loading",
        "model_loaded": ready,
        "version": APP_VERSION,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    classifier = state["classifier"]
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet")

    payload = await file.read()
    if not payload:
        metrics.record(0.0, failed=True)
        raise HTTPException(status_code=400, detail="Empty upload")
    if len(payload) > MAX_UPLOAD_BYTES:
        metrics.record(0.0, failed=True)
        raise HTTPException(status_code=413, detail="Image larger than 8 MB")

    started = time.perf_counter()
    try:
        result = classifier.predict(payload)
    except InvalidImageError as error:
        metrics.record(time.perf_counter() - started, failed=True)
        raise HTTPException(status_code=400, detail=str(error))

    latency = time.perf_counter() - started
    metrics.record(latency, label=result["label"])

    logger.info(
        json.dumps(
            {
                "event": "prediction",
                "filename": file.filename,
                "label": result["label"],
                "confidence": result["confidence"],
                "inference_ms": round(latency * 1000, 2),
            }
        )
    )

    result["inference_ms"] = round(latency * 1000, 2)
    return result


@app.get("/stats")
def stats():
    return metrics.snapshot()


@app.get("/metrics", response_class=PlainTextResponse)
def prometheus_metrics():
    snapshot = metrics.snapshot()
    lines = [
        "# HELP inference_requests_total Prediction requests handled",
        "# TYPE inference_requests_total counter",
        f"inference_requests_total {snapshot['requests_total']}",
        "# HELP inference_errors_total Prediction requests rejected or failed",
        "# TYPE inference_errors_total counter",
        f"inference_errors_total {snapshot['errors_total']}",
        "# HELP inference_latency_ms_avg Average model latency in milliseconds",
        "# TYPE inference_latency_ms_avg gauge",
        f"inference_latency_ms_avg {snapshot['average_latency_ms']}",
        "# HELP predictions_by_class_total Predictions grouped by predicted class",
        "# TYPE predictions_by_class_total counter",
    ]
    for label, count in snapshot["predictions"].items():
        lines.append(f'predictions_by_class_total{{class="{label}"}} {count}')
    return "\n".join(lines) + "\n"
