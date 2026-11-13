import argparse
import csv
from pathlib import Path

import matplotlib
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from deep_learning_pa3.dataset import AnimalDataset
from deep_learning_pa3.evaluation import evaluate_model, load_class_names, save_confusion_report
from deep_learning_pa3.paths import CLASSES_PATH, DATA_DIR, OUTPUT_DIR, RESNET_WEIGHTS_PATH, WEIGHTS_DIR
from deep_learning_pa3.resnet_classifier import ResNetClassifier
from deep_learning_pa3.utils import SquareImageTensor


class BioCLIPClassifier(nn.Module):
    """Return an image-to-prompt similarity for each PA3 class."""

    def __init__(self, model: nn.Module, text_features: Tensor) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("text_features", text_features)

    def forward(self, images: Tensor) -> Tensor:
        image_features = self.model.encode_image(images, normalize=True)
        return image_features @ self.text_features.T


def main() -> None:

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=RESNET_WEIGHTS_PATH)
    parser.add_argument(
        "--unfiltered", type=Path, default=WEIGHTS_DIR / "student_threshold_0.0_seed_42.pt"
    )
    parser.add_argument("--filtered", type=Path, required=True, help="Checkpoint selected on val")
    args = parser.parse_args()

    checkpoints = {
        "baseline": args.baseline,
        "student_unfiltered": args.unfiltered,
        "student_filtered": args.filtered,
    }
    
    import open_clip

    from deep_learning_pa3.teachers_prompts import build_prompts

    matplotlib.use("Agg")
    class_names = load_class_names(CLASSES_PATH)
    num_classes = len(class_names)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resnet_data = AnimalDataset(
        root=DATA_DIR, split="test_public",
        transform=SquareImageTensor(224, imagenet_normalize=True),
    )
    resnet_loader = DataLoader(resnet_data, batch_size=16, shuffle=False)
    results = {}
    for name, path in checkpoints.items():
        print(f"Evaluating {name} on {device}...", flush=True)
        model = ResNetClassifier(num_classes=num_classes, pretrained=False).to(device)
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        results[name] = evaluate_model(model, resnet_loader, device, num_classes)
    del model

    print(f"Evaluating BioCLIP on {device}...", flush=True)
    model_id = "hf-hub:imageomics/bioclip"
    teacher, _, preprocess = open_clip.create_model_and_transforms(model_id, device=device)
    teacher.eval()
    tokenizer = open_clip.get_tokenizer(model_id)
    tokens = tokenizer(build_prompts(class_names)).to(device)
    with torch.inference_mode():
        text_features = teacher.encode_text(tokens, normalize=True)
    teacher_data = AnimalDataset(root=DATA_DIR, split="test_public", transform=preprocess)
    teacher_loader = DataLoader(teacher_data, batch_size=16, shuffle=False)
    teacher_result = evaluate_model(
        BioCLIPClassifier(teacher, text_features), teacher_loader, device, num_classes
    )
    results["bioclip"] = teacher_result

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_DIR / "trilha_c_test_public.csv"
    per_class_path = OUTPUT_DIR / "trilha_c_test_public_by_class.csv"
    class_f1 = {}
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["model", "macro_f1", "accuracy"])
        for name, (confusion, macro_f1) in results.items():
            correct = confusion.diag()
            accuracy = correct.sum().item() / confusion.sum().item()
            support = confusion.sum(dim=1)
            predicted = confusion.sum(dim=0)
            class_f1[name] = (2 * correct / (support + predicted).clamp_min(1)).tolist()
            writer.writerow([name, macro_f1, accuracy])
            print(f"{name}: macro F1={macro_f1:.4f}, accuracy={accuracy:.4f}")

    teacher_confusion, teacher_f1 = teacher_result
    with per_class_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["class", "support", *class_f1, "delta_filtered_vs_bioclip"])
        for index, (label, count) in enumerate(zip(class_names, teacher_confusion.sum(dim=1).tolist())):
            delta = class_f1["student_filtered"][index] - class_f1["bioclip"][index]
            writer.writerow([label, count, *(values[index] for values in class_f1.values()), delta])

    matrix_path, plot_path = save_confusion_report(
        teacher_confusion, class_names, OUTPUT_DIR, "bioclip", teacher_f1
    )
    for path in (summary_path, per_class_path, matrix_path, plot_path):
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
