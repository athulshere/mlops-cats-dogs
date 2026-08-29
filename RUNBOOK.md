# Runbook — running this on your machine

Follow these in order. Total time is roughly 45 minutes, most of it waiting on
training and the first Docker build.

---

## Before you start

Install these if you don't have them:

| Tool | Notes |
|---|---|
| Python 3.11, 3.12 or 3.13 | `python3 -V` to check. 3.10 and below won't work — numpy and scikit-learn dropped it |
| Docker Desktop | **Launch it.** The daemon has to be running, not just installed |
| Git | `git --version` to check |
| VS Code | Plus the **Python** and **Docker** extensions |

You also need a GitHub account and a Kaggle account.

---

## Step 1 — Open the folder

In VS Code: **File → Open Folder →** `mlops-cats-dogs`.

Open the built-in terminal with `` Ctrl+` ``. Every command below runs there.

---

## Step 2 — Virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-dev.txt
```

That torch index URL matters — the default Linux wheel pulls in the entire CUDA
stack, several gigabytes you have no use for on a laptop.

Now tell VS Code about it: `Cmd/Ctrl+Shift+P` → **Python: Select Interpreter** →
choose the entry containing `.venv`. Skip this and every import shows a red
squiggle and the Testing panel stays empty.

Check where you stand:

```bash
python scripts/verify_setup.py
```

Dataset and checkpoint will fail at this point — that's expected. Everything else
should say `[ok]`.

---

## Step 3 — Confirm the code works

```bash
pytest -v
```

16 tests, all passing. They build a throwaway model in a temp folder, so they
need neither the dataset nor a trained checkpoint. **If this fails, it's an
install problem** — sort it out here rather than discovering it later.

You can also run these from the flask icon in the VS Code sidebar.

---

## Step 4 — Dataset

Download the [Kaggle cats vs dogs dataset](https://www.kaggle.com/datasets/salader/dogs-vs-cats)
and unzip it into `data/raw/`.

```bash
find data/raw -name "*.jpg" | wc -l
```

Thousands means you're fine. Zero usually means the unzip nested things one level
deeper than expected — look for an inner folder and move its contents up. The
loader reads both folder names and file names, so `PetImages/Cat/*.jpg`,
`train/cats/*.jpg` and `train/cat.0.jpg` all work.

---

## Step 5 — Train

```bash
python -m src.data_prep      # 2-3 minutes
python -m src.train          # 15-25 minutes on CPU
```

Watch the first epoch. If it's under about 4 minutes, let the whole thing run. If
it's crawling, stop it and run just these two commands in Google Colab on a T4
(around 5 minutes for 8 epochs), then copy `models/cats_dogs_cnn.pt`, the
`reports/` folder and `mlflow.db` back here. The checkpoint loads with
`map_location="cpu"`, so a GPU-trained model runs fine in the container.

Expect 80–88% test accuracy at the 5-epoch default. Much lower, bump
`train.epochs` to 8 in `params.yaml` and run again.

Look at the tracking UI — open a **second terminal** with the `+` button rather
than killing the first:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

`Cmd/Ctrl+click` the localhost link. Confirm the run shows its parameters, the
per-epoch curves, and the confusion matrix under Artifacts. You'll be showing
this on camera.

---

## Step 6 — Commit the checkpoint

```bash
git add -f models/cats_dogs_cnn.pt
git commit -m "Add trained baseline checkpoint"
```

The `-f` is not optional — `.gitignore` would skip the file otherwise. The
Dockerfile copies `models/`, so **forgetting this is the most common reason the
CI build fails.**

---

## Step 7 — Run the API locally

Terminal 1:

```bash
uvicorn app.main:app --reload --port 8000
```

Terminal 2:

```bash
curl http://localhost:8000/health
curl -X POST -F "file=@$(ls data/processed/test/dogs/*.jpg | head -1)" http://localhost:8000/predict
```

Also open `http://localhost:8000/docs` — the Swagger page lets you upload an image
by clicking, which films far better than curl.

Stop uvicorn with `Ctrl+C` before the next step or port 8000 stays occupied.

---

## Step 8 — Container

```bash
docker compose up -d --build
docker compose ps
python scripts/smoke_test.py --url http://localhost:8000
python scripts/monitor_batch.py --url http://localhost:8000 --per-class 25
```

The first build downloads the Python base image plus torch, so give it 5–10
minutes. The Docker panel in the VS Code sidebar shows the container and its logs.

Open `reports/post_deploy_report.json` and compare its `accuracy` against
`reports/metrics.json`. That comparison is your monitoring evidence — live
performance measured against training performance.

---

## Step 9 — GitHub

Create an empty repo on github.com. No README, no .gitignore — you already have
both.

First, edit `k8s/deployment.yaml` line 18. Replace `ghcr.io/OWNER/REPO:latest`
with `ghcr.io/<your-username>/mlops-cats-dogs:latest`, **all lowercase**. Commit
that change.

```bash
git remote add origin https://github.com/<you>/mlops-cats-dogs.git
git branch -M main
git push -u origin main
```

Then in the repo on GitHub: **Settings → Actions → General → Workflow
permissions → Read and write permissions → Save.** Without this the image push
returns a 403.

Open the **Actions** tab. Three jobs run in order — tests, build and push, then
deploy with the smoke test. Green all the way means the CI and CD modules are
demonstrably working. Your image lands under the repo's **Packages** section.

---

## Step 10 — Screen recording

Change `APP_VERSION` in `app/main.py` from `"1.0.0"` to `"1.1.0"`, commit, and
push while recording. That single push is the "code change to deployed
prediction" story the brief asks for.

Rough timings to stay under five minutes:

| Shot | Time |
|---|---|
| MLflow UI — runs, metrics, confusion matrix | 45s |
| `git log`, `dvc.yaml` — versioning | 20s |
| The version bump, commit and push | 30s |
| Actions running (fast-forward the build) | 90s |
| `curl /health` and `/predict` against the deployed container | 45s |
| `curl /metrics`, `tail logs/requests.log` | 30s |
| `python scripts/monitor_batch.py` output | 30s |

---

## Step 11 — Package for submission

```bash
git status                    # nothing uncommitted
docker compose down
cd ..
zip -r mlops-cats-dogs-submission.zip mlops-cats-dogs \
  -x "*/.venv/*" "*/data/raw/*" "*/data/processed/*" "*/__pycache__/*" "*/.pytest_cache/*"
```

This keeps the trained model, the report images, `metrics.json`,
`post_deploy_report.json` and `mlflow.db` — all of which are your evidence — and
drops the image folders and the virtual environment.

---

## When something breaks

**`ModuleNotFoundError: No module named 'src'`**
You're not in the project root, or the venv isn't active. `cd` to the folder with
`params.yaml` in it and re-activate.

**Imports underlined red in VS Code, but the terminal works fine**
Interpreter isn't selected. `Cmd/Ctrl+Shift+P` → Python: Select Interpreter →
pick `.venv`.

**`Cannot connect to the Docker daemon`**
Docker Desktop isn't running. Launch it, wait for the whale icon to settle.

**`Error response from daemon: ... 403` on the CI push**
Workflow permissions in step 9.

**CI build fails at `COPY models/`**
The checkpoint wasn't committed. Redo step 6.

**Port 8000 already in use**
Something's still bound to it. `lsof -ti:8000 | xargs kill` on macOS/Linux, or
`docker compose down` if it's the container.

**`Could not find a version that satisfies the requirement torch==X`**
The pinned torch build has aged out of the CPU wheel index. Look at the versions
pip lists in the error, pick the newest, and set torch plus its matching
torchvision in `requirements.txt` — the pairs go 2.11/0.26, 2.12/0.27, 2.13/0.28.

**Training accuracy stuck near 50%**
The model is guessing, which usually means only one class made it through data
prep. Re-run `python -m src.data_prep` and check the printed counts — you should
see roughly equal cats and dogs in every split.
