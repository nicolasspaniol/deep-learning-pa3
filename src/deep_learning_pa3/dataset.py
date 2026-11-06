"""Dataset utilities for the PA3 animal image classification data."""

import csv
from pathlib import Path
from typing import Any, Callable, Literal

from PIL import Image
from torch.utils.data import Dataset


Split = Literal["train", "val", "test_public"]


class AnimalDataset(Dataset[tuple[Any, int]]):
    """Load labeled images from the repository's ``data`` directory.

    Training data uses one of the labeled subsets in ``labels/rotulos_K.csv``;
    validation and public test data use their corresponding label files.
    Images are returned as RGB PIL images unless ``transform`` is provided.

    Args:
        root: Dataset directory (defaults to ``./data`` relative to the
            current working directory).
        split: One of ``"train"``, ``"val"``, or ``"test_public"``.
        label_budget: Number of labels per class to use for training. Ignored
            for validation and test splits.
        transform: Optional transform applied to each image.
    """

    def __init__(
        self,
        root: str | Path = "./data",
        split: Split = "train",
        label_budget: Literal[1, 5, 20] = 20,
        transform: Callable[[Image.Image], Any] | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.transform = transform

        if split not in ("train", "val", "test_public"):
            raise ValueError(f"Unsupported split {split!r}")
        if split == "train" and label_budget not in (1, 5, 20):
            raise ValueError("label_budget must be 1, 5, or 20")

        labels_file = (
            self.root / "labels" / f"rotulos_{label_budget}.csv"
            if split == "train"
            else self.root / "labels" / f"{split}.csv"
        )
        images_dir = self.root / "images" / split

        with (self.root / "classes.csv").open(newline="", encoding="utf-8") as file:
            classes = {row["label"]: int(row["idx"]) for row in csv.DictReader(file)}

        self.samples: list[tuple[Path, int]] = []
        with labels_file.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                image_id = row["image_id"]
                label = row["label"]
                self.samples.append((images_dir / f"{image_id}.jpg", classes[label]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        image_path, label = self.samples[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label
