from pathlib import Path
from random import randint

import matplotlib.pyplot as plt

from deep_learning_pa3.dataset import AnimalDataset


data_dir = Path(__file__).resolve().parents[1] / "data"
dataset = AnimalDataset(root=data_dir, split="val")
image, label = dataset[randint(0, 100)]

print(f"Loaded {len(dataset)} validation images; sample label index: {label}")
plt.imshow(image)
plt.title(f"Class index: {label}")
plt.axis("off")
plt.show()
