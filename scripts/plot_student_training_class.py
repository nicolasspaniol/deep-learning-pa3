"""Plot student training images predicted as a selected class."""

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch.utils.data import DataLoader

from deep_learning_pa3.dataset import AnimalDataset
from deep_learning_pa3.evaluation import load_class_names
from deep_learning_pa3.paths import CLASSES_PATH, DATA_DIR, OUTPUT_DIR, WEIGHTS_DIR
from deep_learning_pa3.resnet_classifier import ResNetClassifier
from deep_learning_pa3.train_student import add_pseudo_labels
from deep_learning_pa3.utils import SquareImageTensor


DEFAULT_CHECKPOINT = WEIGHTS_DIR / "student_threshold_0.8_seed_42.pt"
DEFAULT_PSEUDO_LABELS = OUTPUT_DIR / "teacher_train.csv"
IMAGE_SIZE = 224


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--class-name", default="eastern_chipmunk")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--pseudo-labels", type=Path, default=DEFAULT_PSEUDO_LABELS)
    parser.add_argument(
        "--pseudo-threshold",
        type=float,
        default=0.8,
        help="Confidence threshold used to reconstruct this student's training set",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR / "student_training_eastern_chipmunk_predictions.png",
    )
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        parser.error(f"Student checkpoint not found: {args.checkpoint}")
    if not args.pseudo_labels.is_file():
        parser.error(f"Pseudo-label file not found: {args.pseudo_labels}")
    if not 0 <= args.pseudo_threshold <= 1:
        parser.error("--pseudo-threshold must be between 0 and 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")

    class_names = load_class_names(CLASSES_PATH)
    if args.class_name not in class_names:
        parser.error(
            f"Unknown class {args.class_name!r}; choose from: {', '.join(class_names)}"
        )
    target_index = class_names.index(args.class_name)

    transform = SquareImageTensor(IMAGE_SIZE, imagenet_normalize=True)
    dataset = AnimalDataset(
        root=DATA_DIR,
        split="train",
        label_budget=5,
        transform=transform,
    )
    manual_ids = {path.stem for path, _ in dataset.samples}
    pseudo_counts = add_pseudo_labels(
        dataset,
        args.pseudo_labels,
        class_names,
        args.pseudo_threshold,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNetClassifier(num_classes=len(class_names), pretrained=False).to(device)
    model.load_state_dict(
        torch.load(args.checkpoint, map_location=device, weights_only=True)
    )
    model.eval()

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    predictions: list[tuple[int, float]] = []
    with torch.inference_mode():
        for batch_index, (images, _labels) in enumerate(loader):
            probabilities = model(images.to(device)).softmax(dim=1).cpu()
            predicted_classes = probabilities.argmax(dim=1)
            batch_start = batch_index * args.batch_size
            for offset, predicted_index in enumerate(predicted_classes.tolist()):
                if predicted_index == target_index:
                    predictions.append(
                        (batch_start + offset, probabilities[offset, target_index].item())
                    )

    print(
        f"Student predicted {args.class_name} for {len(predictions)} of "
        f"{len(dataset)} training images."
    )
    print(
        f"Training set: {len(manual_ids)} manually labeled images and "
        f"{len(dataset) - len(manual_ids)} accepted pseudo-labels "
        f"(threshold {args.pseudo_threshold:g}; "
        f"{pseudo_counts[args.class_name]} pseudo-labeled as {args.class_name})."
    )

    if predictions:
        columns = min(4, len(predictions))
        rows = math.ceil(len(predictions) / columns)
        figure, axes = plt.subplots(
            rows,
            columns,
            figsize=(4.5 * columns, 4 * rows),
            squeeze=False,
        )
        for axis in axes.flat:
            axis.axis("off")

        for axis, (index, confidence) in zip(axes.flat, predictions):
            image_path, training_label = dataset.samples[index]
            with Image.open(image_path) as image:
                axis.imshow(image.convert("RGB"))
            source = "manual" if image_path.stem in manual_ids else "pseudo"
            axis.set_title(
                f"ID: {image_path.stem}\n"
                f"Student: {args.class_name} ({confidence:.1%})\n"
                f"Train label: {class_names[training_label]} ({source})"
            )
            axis.axis("off")
            print(
                f"{image_path.stem} | confidence={confidence:.1%} | "
                f"train label={class_names[training_label]} ({source})"
            )
        figure.suptitle(
            f"Training images predicted as {args.class_name} "
            f"({args.checkpoint.name})"
        )
        figure.tight_layout()
    else:
        figure, axis = plt.subplots(figsize=(8, 3))
        axis.text(0.5, 0.5, f"No images predicted as {args.class_name}", ha="center")
        axis.axis("off")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {args.output}")
    plt.show()


if __name__ == "__main__":
    main()
