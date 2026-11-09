"""Train the scratch CNN on the 5-label-per-class training subset."""

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from deep_learning_pa3.cnn_classifier import CNNClassifier
from deep_learning_pa3.dataset import AnimalDataset
from deep_learning_pa3.evaluation import evaluate_model
from deep_learning_pa3.paths import (
    CNN_WEIGHTS_PATH,
    DATA_DIR,
    OUTPUT_DIR,
    WEIGHTS_DIR,
)
from deep_learning_pa3.utils import SquareImageTensor


EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
IMAGE_SIZE = 128
RANDOM_SEED = 42
LABEL_BUDGET = 5
NUM_CLASSES = 21


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = AnimalDataset(
        root=DATA_DIR,
        split="train",
        label_budget=LABEL_BUDGET,
        transform=SquareImageTensor(IMAGE_SIZE),
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    train_eval_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    validation_dataset = AnimalDataset(
        root=DATA_DIR,
        split="val",
        transform=SquareImageTensor(IMAGE_SIZE),
    )
    validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = CNNClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_f1_by_epoch: list[float] = []
    validation_f1_by_epoch: list[float] = []

    for epoch in range(EPOCHS):
        model.train()
        progress = tqdm(loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

        for images, labels in progress:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            progress.set_postfix(loss=f"{loss.item():.3f}")

        _, train_f1 = evaluate_model(model, train_eval_loader, device, NUM_CLASSES)
        _, validation_f1 = evaluate_model(model, validation_loader, device, NUM_CLASSES)
        train_f1_by_epoch.append(train_f1)
        validation_f1_by_epoch.append(validation_f1)
        print(
            f"Epoch {epoch + 1}: train macro F1={train_f1:.4f}, "
            f"validation macro F1={validation_f1:.4f}"
        )

    torch.save(model.cpu().state_dict(), CNN_WEIGHTS_PATH)

    epochs = range(1, EPOCHS + 1)
    plt.plot(epochs, train_f1_by_epoch, marker="o", label="Train")
    plt.plot(epochs, validation_f1_by_epoch, marker="o", label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Macro F1")
    plt.title("CNN macro F1 (105 labeled training images)")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plot_path = OUTPUT_DIR / "cnn_training_validation_f1.png"
    plt.savefig(plot_path)
    print(f"Saved model to {CNN_WEIGHTS_PATH}")
    print(f"Saved F1 plot to {plot_path}")
    plt.show()


if __name__ == "__main__":
    main()
