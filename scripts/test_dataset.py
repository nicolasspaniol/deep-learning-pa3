from random import randint

import matplotlib.pyplot as plt

from deep_learning_pa3.dataset import AnimalDataset
from deep_learning_pa3.paths import DATA_DIR


dataset = AnimalDataset(root=DATA_DIR, split="val")
image, label = dataset[randint(0, len(dataset))]

print(f"Loaded {len(dataset)} validation images; sample label index: {label}")
plt.imshow(image)
plt.title(f"Class index: {label}")
plt.axis("off")
plt.show()
