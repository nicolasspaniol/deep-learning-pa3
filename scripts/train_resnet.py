"""Train the ResNet-18 classifier's FC layer on rotulos_5."""

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from deep_learning_pa3.dataset import AnimalDataset
from deep_learning_pa3.paths import (
    DATA_DIR,
    OUTPUT_DIR,
    RESNET_WEIGHTS_PATH,
    WEIGHTS_DIR,
)
from deep_learning_pa3.resnet_classifier import ResNetClassifier
from deep_learning_pa3.utils import SquareImageTensor


EPOCHS = 20
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
IMAGE_SIZE = 224
RANDOM_SEED = 42
LABEL_BUDGET = 5

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = AnimalDataset(
        root=DATA_DIR,
        split="train",
        label_budget=LABEL_BUDGET,
        transform=SquareImageTensor(IMAGE_SIZE, imagenet_normalize=True),
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = ResNetClassifier(pretrained=True).to(device)
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False
    for parameter in model.backbone.fc.parameters():
        parameter.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.backbone.fc.parameters(), lr=LEARNING_RATE)
    accuracy_by_epoch: list[float] = []

    for epoch in range(EPOCHS):
        model.train()
        model.backbone.eval()  # Keep frozen batch-normalization statistics fixed.
        model.backbone.fc.train()
        correct = 0
        total = 0

        progress = tqdm(loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
        for images, labels in progress:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
            progress.set_postfix(loss=f"{loss.item():.3f}", accuracy=f"{correct / total:.1%}")

        accuracy_by_epoch.append(correct / total)

    torch.save(model.cpu().state_dict(), RESNET_WEIGHTS_PATH)

    plt.plot(range(1, EPOCHS + 1), accuracy_by_epoch, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Training accuracy")
    plt.title("ResNet-18 FC training accuracy (105 labeled images)")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = OUTPUT_DIR / "resnet_training_accuracy.png"
    plt.savefig(plot_path)
    print(f"Saved model to {RESNET_WEIGHTS_PATH}")
    print(f"Saved accuracy plot to {plot_path}")
    plt.show()


if __name__ == "__main__":
    main()
