from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARAMS_FILE = PROJECT_ROOT / "params.yaml"


def load_params(path=None):
    with open(path or PARAMS_FILE) as handle:
        return yaml.safe_load(handle)


def resolve(relative_path):
    """Paths in params.yaml are relative to the repo root, not the caller's cwd."""
    path = Path(relative_path)
    return path if path.is_absolute() else PROJECT_ROOT / path
