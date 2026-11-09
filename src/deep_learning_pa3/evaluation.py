"""Shared classification evaluation helpers."""

import csv
from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import Tensor, nn


def evaluate_model(
    model: nn.Module,
    batches: Iterable[tuple[Tensor, Tensor]],
    device: torch.device,
    num_classes: int,
) -> tuple[Tensor, float]:
    """Return the rows=true confusion matrix and macro F1 for labeled batches."""
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    model.eval()
    with torch.inference_mode():
        for images, labels in batches:
            predictions = model(images.to(device)).argmax(dim=1).cpu()
            labels = labels.cpu()
            confusion += torch.bincount(
                labels * num_classes + predictions,
                minlength=num_classes * num_classes,
            ).reshape(num_classes, num_classes)

    true_positives = confusion.diag().float()
    false_positives = confusion.sum(dim=0).float() - true_positives
    false_negatives = confusion.sum(dim=1).float() - true_positives
    f1_by_class = 2 * true_positives / (
        2 * true_positives + false_positives + false_negatives
    ).clamp_min(1)
    return confusion, f1_by_class.mean().item()


def load_class_names(classes_file: Path) -> list[str]:
    with classes_file.open(newline="", encoding="utf-8") as file:
        classes = sorted(csv.DictReader(file), key=lambda row: int(row["idx"]))
    return [row["label"] for row in classes]


def save_confusion_report(
    confusion: Tensor,
    class_names: list[str],
    output_dir: Path,
    model_name: str,
    macro_f1: float,
) -> tuple[Path, Path]:
    """Save a labeled confusion-matrix CSV and plot; return both paths."""
    output_dir.mkdir(exist_ok=True)
    stem = f"{model_name}_test_public_confusion_matrix"
    matrix_path = output_dir / f"{stem}.csv"
    with matrix_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["true/predicted", *class_names])
        writer.writerows(
            [class_names[index], *row.tolist()]
            for index, row in enumerate(confusion)
        )

    figure, axis = plt.subplots(figsize=(14, 12))
    image = axis.imshow(confusion.tolist(), cmap="Blues")
    axis.set_xticks(range(len(class_names)), labels=class_names, rotation=90)
    axis.set_yticks(range(len(class_names)), labels=class_names)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title(f"{model_name} test_public confusion matrix (macro F1: {macro_f1:.4f})")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()

    plot_path = output_dir / f"{stem}.png"
    figure.savefig(plot_path)
    plt.show()
    return matrix_path, plot_path
