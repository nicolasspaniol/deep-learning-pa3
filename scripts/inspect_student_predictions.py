"""Show random test_public images with the student's predictions and scores."""

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageOps

from deep_learning_pa3.dataset import AnimalDataset
from deep_learning_pa3.evaluation import load_class_names
from deep_learning_pa3.grad_cam import GradCAM
from deep_learning_pa3.occlusion import Occlusion
from deep_learning_pa3.paths import CLASSES_PATH, DATA_DIR, WEIGHTS_DIR
from deep_learning_pa3.resnet_classifier import ResNetClassifier
from deep_learning_pa3.utils import SquareImageTensor


IMAGE_SIZE = 224
DEFAULT_CHECKPOINT = WEIGHTS_DIR / "checkpoint"


def content_bounds(image_path: Path) -> tuple[int, int, int, int]:
    """Return the non-padding box in the square model input."""
    with Image.open(image_path) as image:
        content_size = ImageOps.contain(
            image,
            (IMAGE_SIZE, IMAGE_SIZE),
            method=Image.Resampling.BILINEAR,
        ).size

    left = round((IMAGE_SIZE - content_size[0]) / 2)
    top = round((IMAGE_SIZE - content_size[1]) / 2)
    return left, top, left + content_size[0], top + content_size[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=6, help="Number of random images to show")
    parser.add_argument("--seed", type=int, default=None, help="Seed for repeatable sampling")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--top-k", type=int, default=5, help="Number of class scores to print")
    parser.add_argument(
        "--method",
        choices=("occlusion", "gradcam"),
        default="gradcam",
        help="Attribution method to visualize",
    )
    parser.add_argument(
        "--overlay-alpha",
        "--cam-alpha",
        dest="overlay_alpha",
        type=float,
        default=0.45,
        help="Opacity of the attribution overlay",
    )
    parser.add_argument(
        "--occlusion-window", type=int, default=32, help="Occlusion patch width and height"
    )
    parser.add_argument("--occlusion-stride", type=int, default=16)
    args = parser.parse_args()

    if args.count < 1:
        parser.error("--count must be at least 1")
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")
    if not 0 <= args.overlay_alpha <= 1:
        parser.error("--overlay-alpha must be between 0 and 1")
    if args.occlusion_window < 1 or args.occlusion_stride < 1:
        parser.error("--occlusion-window and --occlusion-stride must be positive")
    if not args.checkpoint.is_file():
        parser.error(f"Student checkpoint not found: {args.checkpoint}")

    transform = SquareImageTensor(IMAGE_SIZE, imagenet_normalize=True)
    dataset = AnimalDataset(root=DATA_DIR, split="test_public", transform=transform)
    if args.count > len(dataset):
        parser.error(f"--count cannot exceed the {len(dataset)} images in test_public")

    class_names = load_class_names(CLASSES_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNetClassifier(num_classes=len(class_names), pretrained=False).to(device)
    model.load_state_dict(
        torch.load(args.checkpoint, map_location=device, weights_only=True)
    )
    model.eval()
    if args.method == "gradcam":
        explainer: GradCAM | Occlusion = GradCAM(model, model.backbone.layer4[-1])
    else:
        explainer = Occlusion(
            model,
            window_size=args.occlusion_window,
            stride=args.occlusion_stride,
        )

    rng = random.Random(args.seed)
    indices = rng.sample(range(len(dataset)), args.count)
    columns = min(3, args.count)
    rows = (args.count + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(5 * columns, 4.5 * rows), squeeze=False)

    for axis, index in zip(axes.flat, indices):
        image, true_index = dataset[index]
        image_path = dataset.samples[index][0]
        image_id = image_path.stem
        heatmap, logits = explainer.generate(image.unsqueeze(0).to(device))
        probabilities = logits.softmax(dim=0)
        predicted_index = int(probabilities.argmax())
        top_scores = probabilities.topk(min(args.top_k, len(class_names)))

        # Remove letterbox padding, which is part of the model input but not
        # part of the photographed scene, from both the image and the overlay.
        left, top, right, bottom = content_bounds(image_path)
        display_image = (image * transform.std + transform.mean).clamp(0, 1)
        display_image = display_image[:, top:bottom, left:right]
        heatmap = heatmap[top:bottom, left:right]
        heatmap = heatmap - heatmap.min()
        heatmap = heatmap / heatmap.max().clamp_min(1e-8)
        axis.imshow(display_image.permute(1, 2, 0))
        axis.imshow(heatmap, cmap="jet", alpha=args.overlay_alpha, vmin=0, vmax=1)
        axis.set_title(
            f"ID: {image_id}\n"
            f"True: {class_names[true_index]} | "
            f"Pred: {class_names[predicted_index]} "
            f"({probabilities[predicted_index].item():.1%})"
        )
        axis.axis("off")

        scores = ", ".join(
            f"{class_names[class_index]}={score:.1%}"
            for score, class_index in zip(
                top_scores.values.tolist(), top_scores.indices.tolist()
            )
        )
        print(
            f"ID: {image_id} | true: {class_names[true_index]} | "
            f"pred: {class_names[predicted_index]} | top scores: {scores}"
        )

    for axis in axes.flat[args.count :]:
        axis.axis("off")
    figure.suptitle(f"Student {args.method} on test_public ({args.checkpoint.name})")
    figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
