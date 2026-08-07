from PIL import Image

from src.data_prep import collect_images, label_for, preprocess_image, split_items


def make_image(path, size=(80, 120), mode="RGB"):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new(mode, size, color=(120, 60, 30)).save(path)
    return path


def test_label_is_read_from_folder_or_filename():
    assert label_for("data/raw/PetImages/Cat/45.jpg") == "cats"
    assert label_for("data/raw/train/dog.1024.jpg") == "dogs"
    assert label_for("data/raw/misc/readme.txt") is None


def test_collect_images_ignores_unlabelled_files(tmp_path):
    make_image(tmp_path / "Cat" / "a.jpg")
    make_image(tmp_path / "Dog" / "b.jpg")
    make_image(tmp_path / "other" / "c.jpg")
    (tmp_path / "Cat" / "notes.txt").write_text("ignore me")

    labels = sorted(label for _, label in collect_images(tmp_path))

    assert labels == ["cats", "dogs"]


def test_preprocess_image_returns_square_rgb():
    grayscale = Image.new("L", (300, 150))
    processed = preprocess_image(grayscale, size=224)

    assert processed.size == (224, 224)
    assert processed.mode == "RGB"


def test_split_respects_ratios_and_keeps_both_classes():
    items = [(f"cat_{i}.jpg", "cats") for i in range(100)]
    items += [(f"dog_{i}.jpg", "dogs") for i in range(100)]

    buckets = split_items(items, train_ratio=0.8, val_ratio=0.1, seed=1)

    assert len(buckets["train"]) == 160
    assert len(buckets["val"]) == 20
    assert len(buckets["test"]) == 20
    for split in buckets.values():
        assert {label for _, label in split} == {"cats", "dogs"}


def test_splits_do_not_overlap_and_are_reproducible():
    items = [(f"cat_{i}.jpg", "cats") for i in range(50)]
    items += [(f"dog_{i}.jpg", "dogs") for i in range(50)]

    first = split_items(items, 0.8, 0.1, seed=42)
    second = split_items(items, 0.8, 0.1, seed=42)

    assert first["train"] == second["train"]
    train_files = {path for path, _ in first["train"]}
    test_files = {path for path, _ in first["test"]}
    assert train_files.isdisjoint(test_files)


def test_limit_per_class_caps_the_sample_size():
    items = [(f"cat_{i}.jpg", "cats") for i in range(500)]
    items += [(f"dog_{i}.jpg", "dogs") for i in range(500)]

    buckets = split_items(items, 0.8, 0.1, seed=7, limit_per_class=50)
    total = sum(len(split) for split in buckets.values())

    assert total == 100
