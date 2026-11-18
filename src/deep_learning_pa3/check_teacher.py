import argparse
import random

from deep_learning_pa3.paths import CLASSES_PATH, DATA_DIR, OUTPUT_DIR
from deep_learning_pa3.teachers_prompts import build_prompts


MODEL_ID = "hf-hub:imageomics/bioclip"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-images", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    if args.num_images < 1:
        parser.error("--num-images must be at least 1")
    if not CLASSES_PATH.is_file():
        parser.error(f"Missing class names: {CLASSES_PATH}")

    train_dir = DATA_DIR / "images" / "train"
    image_paths = sorted(train_dir.glob("*.jpg"))
    if not image_paths:
        parser.error(f"No .jpg training images found in {train_dir}")

    sample_size = min(args.num_images, len(image_paths))
    selected_paths = random.Random(args.seed).sample(image_paths, sample_size)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import open_clip
    import torch
    from PIL import Image

    from deep_learning_pa3.evaluation import load_class_names

    class_names = load_class_names(CLASSES_PATH)
    if len(class_names) != 21:
        parser.error(f"Expected 21 PA3 classes, found {len(class_names)}")
    prompts = build_prompts(class_names)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable. Use --device cpu or --device auto")

    print(f"Loading {MODEL_ID} on {device}...", flush=True)
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_ID, device=device
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(MODEL_ID)
    tokens = tokenizer(prompts)
    if not isinstance(tokens, torch.Tensor):
        raise TypeError("BioCLIP tokenizer did not return a tensor")
    tokens = tokens.to(device)

    columns = min(3, sample_size)
    rows = (sample_size + columns - 1) // columns
    figure, axes = plt.subplots(
        rows, columns, figsize=(5 * columns, 4 * rows), squeeze=False
    )
    for axis in axes.flat:
        axis.axis("off")

    with torch.inference_mode():
        text_features = model.encode_text(tokens, normalize=True)

        for image_path, axis in zip(selected_paths, axes.flat):
            with Image.open(image_path) as image:
                rgb_image = image.convert("RGB")
            image_tensor = preprocess(rgb_image).unsqueeze(0).to(device)

            image_features = model.encode_image(image_tensor, normalize=True)
            similarities = image_features @ text_features.T
            logits = model.logit_scale.exp() * similarities
            probabilities = logits.softmax(dim=-1)
            score, prediction = probabilities[0].max(dim=-1)

            label = class_names[prediction.item()]
            axis.imshow(rgb_image)
            axis.set_title(f"{label}\nScore: {score.item():.4f}")

    figure.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "teacher_check.png"
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    print(f"Saved predictions to {output_path}")


if __name__ == "__main__":
    main()
