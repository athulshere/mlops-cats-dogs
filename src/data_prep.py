"""Turns the raw Kaggle download into a clean 224x224 train/val/test tree."""

import argparse
import random
import shutil
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from src.config import load_params, resolve

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
CLASSES = ("cats", "dogs")
SPLITS = ("train", "val", "test")


def label_for(path):
    """The Kaggle archive comes in a few different layouts (PetImages/Cat,
    train/cat.123.jpg, ...), so both the folder name and the file name are
    worth checking before we give up on a file."""
    haystack = " ".join(part.lower() for part in Path(path).parts[-3:])
    if "cat" in haystack:
        return "cats"
    if "dog" in haystack:
        return "dogs"
    return None


def collect_images(raw_dir):
    found = []
    for path in sorted(Path(raw_dir).rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label = label_for(path)
        if label is not None:
            found.append((path, label))
    return found


def split_items(items, train_ratio, val_ratio, seed=42, limit_per_class=0):
    """Split per class so train/val/test all stay balanced."""
    rng = random.Random(seed)
    buckets = {name: [] for name in SPLITS}

    for label in CLASSES:
        subset = [item for item in items if item[1] == label]
        rng.shuffle(subset)
        if limit_per_class:
            subset = subset[:limit_per_class]

        total = len(subset)
        n_train = int(total * train_ratio)
        n_val = int(total * val_ratio)

        buckets["train"].extend(subset[:n_train])
        buckets["val"].extend(subset[n_train:n_train + n_val])
        buckets["test"].extend(subset[n_train + n_val:])

    for name in SPLITS:
        rng.shuffle(buckets[name])
    return buckets


def preprocess_image(image, size=224):
    """Everything the CNN sees is RGB and square, whatever came in."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image.resize((size, size), Image.BILINEAR)


def write_split(buckets, processed_dir, image_size):
    processed_dir = Path(processed_dir)
    if processed_dir.exists():
        shutil.rmtree(processed_dir)

    for split in SPLITS:
        for label in CLASSES:
            (processed_dir / split / label).mkdir(parents=True, exist_ok=True)

    counts = {split: {label: 0 for label in CLASSES} for split in SPLITS}
    skipped = 0

    for split, items in buckets.items():
        for index, (source, label) in enumerate(items):
            try:
                with Image.open(source) as image:
                    resized = preprocess_image(image, image_size)
                    target = processed_dir / split / label / f"{label}_{index:05d}.jpg"
                    resized.save(target, "JPEG", quality=90)
            except (UnidentifiedImageError, OSError):
                # A handful of files in the Kaggle set are truncated or not images at all.
                skipped += 1
                continue
            counts[split][label] += 1

    return counts, skipped


def main():
    parser = argparse.ArgumentParser(description="Prepare the cats vs dogs dataset")
    parser.add_argument("--raw-dir")
    parser.add_argument("--processed-dir")
    parser.add_argument("--limit-per-class", type=int)
    args = parser.parse_args()

    params = load_params()["data"]
    raw_dir = resolve(args.raw_dir or params["raw_dir"])
    processed_dir = resolve(args.processed_dir or params["processed_dir"])
    limit = args.limit_per_class if args.limit_per_class is not None else params["limit_per_class"]

    items = collect_images(raw_dir)
    if not items:
        raise SystemExit(
            f"No cat/dog images found under {raw_dir}. "
            "Download the Kaggle dataset and unzip it there first."
        )

    buckets = split_items(
        items,
        train_ratio=params["train_ratio"],
        val_ratio=params["val_ratio"],
        seed=params["seed"],
        limit_per_class=limit,
    )
    counts, skipped = write_split(buckets, processed_dir, params["image_size"])

    print(f"Found {len(items)} labelled images under {raw_dir}")
    for split in SPLITS:
        line = ", ".join(f"{label}={counts[split][label]}" for label in CLASSES)
        print(f"  {split:5s} -> {line}")
    if skipped:
        print(f"Skipped {skipped} unreadable files")


if __name__ == "__main__":
    main()
