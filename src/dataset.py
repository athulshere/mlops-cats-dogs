from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ImageNet statistics - standard choice and it keeps the inputs roughly centred.
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def build_transforms(image_size=224, augment=False):
    steps = [transforms.Resize((image_size, image_size))]
    if augment:
        steps += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(12),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ]
    steps += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    return transforms.Compose(steps)


def build_loaders(processed_dir, image_size=224, batch_size=32, num_workers=2):
    processed_dir = Path(processed_dir)
    loaders = {}
    class_names = None

    for split in ("train", "val", "test"):
        dataset = datasets.ImageFolder(
            processed_dir / split,
            transform=build_transforms(image_size, augment=(split == "train")),
        )
        class_names = dataset.classes
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
        )

    return loaders, class_names
