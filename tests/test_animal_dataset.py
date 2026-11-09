import pytest
from PIL import Image

from deep_learning_pa3.dataset import AnimalDataset, Split
from deep_learning_pa3.paths import DATA_DIR


@pytest.mark.parametrize(
    ("split", "expected_length"),
    [("train", 105), ("val", 202), ("test_public", 811)],
)
def test_dataset_split_lengths(split: Split, expected_length: int) -> None:
    if split == "train":
        dataset = AnimalDataset(root=DATA_DIR, split=split, label_budget=5)
    else:
        dataset = AnimalDataset(root=DATA_DIR, split=split)

    assert len(dataset) == expected_length


def test_dataset_loads_rgb_image_and_class_index() -> None:
    image, label = AnimalDataset(root=DATA_DIR, split="val")[0]

    assert isinstance(image, Image.Image)
    assert image.mode == "RGB"
    assert image.width > 0 and image.height > 0
    assert 0 <= label < 21


def test_dataset_applies_transform() -> None:
    image_size, _ = AnimalDataset(
        root=DATA_DIR,
        split="val",
        transform=lambda image: image.size,
    )[0]

    assert image_size[0] > 0 and image_size[1] > 0
