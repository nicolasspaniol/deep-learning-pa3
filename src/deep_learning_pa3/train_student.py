import argparse
import csv
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from deep_learning_pa3.dataset import AnimalDataset
from deep_learning_pa3.evaluation import evaluate_model, load_class_names
from deep_learning_pa3.paths import DATA_DIR, OUTPUT_DIR, WEIGHTS_DIR
from deep_learning_pa3.resnet_classifier import ResNetClassifier
from deep_learning_pa3.utils import SquareImageTensor

def add_pseudo_labels(dataset: AnimalDataset, csv_path: Path, class_names: list[str], threshold: float) -> Counter[str]:
    
    real_ids = {path.stem for path, _ in dataset.samples}
    class_indices = {label: index for index, label in enumerate(class_names)}
    train_dir = dataset.root / "images" / "train"
    accepted = Counter()
    seen_ids = set()

    with csv_path.open(newline="", encoding="utf-8") as file:

        for row in csv.DictReader(file):
            image_id = row["image_id"]
            if image_id in real_ids:
                continue
            if image_id in seen_ids:
                raise ValueError(f"Duplicate image_id in {csv_path}: {image_id}")
            seen_ids.add(image_id)

            confidence = float(row["confidence"])
            if not 0 <= confidence <= 1:
                raise ValueError(f"Invalid confidence for {image_id}: {confidence}")
            if confidence < threshold:
                continue

            image_path = train_dir / f"{image_id}.jpg"
            if image_path.parent != train_dir or not image_path.is_file():
                raise FileNotFoundError(f"Expected a train image for {image_id}: {image_path}")
            label = row["label"]
            dataset.samples.append((image_path, class_indices[label]))
            accepted[label] += 1

    return accepted

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pseudo-labels", type=Path, default=OUTPUT_DIR / "teacher_train.csv")
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = SquareImageTensor(224, imagenet_normalize=True)

    class_names = load_class_names(DATA_DIR / "classes.csv")
    train = AnimalDataset(root=DATA_DIR, split="train", label_budget=5, transform=transform)

    validation = AnimalDataset(root=DATA_DIR, split="val", transform=transform)
    real_count = len(train)

    accepted = add_pseudo_labels(train, args.pseudo_labels, class_names, args.threshold)
    
    print(f"Device: {device}. Real labels: {real_count}. Pseudo-labels: {accepted.total()}.")
    for label in class_names:
        print(f"  {label}: {accepted[label]}")

    train_loader = DataLoader(train, batch_size=16, shuffle=True)
    validation_loader = DataLoader(validation, batch_size=16, shuffle=False)

    model = ResNetClassifier(num_classes=len(class_names), pretrained=True).to(device)
    model.backbone.requires_grad_(False)
    model.backbone.fc.requires_grad_(True)
    optimizer = torch.optim.Adam(model.backbone.fc.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = WEIGHTS_DIR / f"student_threshold_{args.threshold}_seed_{args.seed}.pt"
    best_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        model.backbone.eval()
        model.backbone.fc.train()
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for images, labels in progress:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            progress.set_postfix(loss=f"{loss.item():.3f}")

        _, validation_f1 = evaluate_model(model, validation_loader, device, len(class_names))
        print(f"Epoch {epoch}: validation macro F1={validation_f1:.4f}")
        if validation_f1 > best_f1:
            best_f1 = validation_f1
            state = {name: value.cpu() for name, value in model.state_dict().items()}
            torch.save(state, checkpoint)

    print(f"Best validation macro F1: {best_f1:.4f}. Checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
