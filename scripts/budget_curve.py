import argparse
import csv
from pathlib import Path
from statistics import mean, stdev
from typing import Literal

import matplotlib
import torch
from torch import nn
from torch.utils.data import DataLoader

from deep_learning_pa3.dataset import AnimalDataset
from deep_learning_pa3.evaluation import evaluate_model, load_class_names
from deep_learning_pa3.paths import DATA_DIR, OUTPUT_DIR
from deep_learning_pa3.resnet_classifier import ResNetClassifier
from deep_learning_pa3.train_student import add_pseudo_labels
from deep_learning_pa3.utils import SquareImageTensor


BUDGET_SEEDS: dict[Literal[1, 5, 20], tuple[int, ...]] = {
    1: (42,),
    5: (42, 43, 44),
    20: (42,),
}
METHODS = ("baseline", "student")
TEACHER_F1 = 0.2636530101299286  # BioCLIP zero-shot on test_public, from Parte 1.
SUPERVISED_REFERENCE = 0.686


def train_checkpoint(train, validation, checkpoint, seed, epochs, num_classes, device):
    torch.manual_seed(seed)
    loader = DataLoader(train, batch_size=16, shuffle=True)
    model = ResNetClassifier(num_classes=num_classes, pretrained=True).to(device)
    model.backbone.requires_grad_(False)
    model.backbone.fc.requires_grad_(True)
    optimizer = torch.optim.Adam(model.backbone.fc.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    best_f1 = -1.0

    for epoch in range(1, epochs + 1):
        model.backbone.eval()
        model.backbone.fc.train()
        for images, labels in loader:
            optimizer.zero_grad()
            loss = criterion(model(images.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()

        _, f1 = evaluate_model(model, validation, device, num_classes)
        print(f"  Epoch {epoch}/{epochs}: val macro F1={f1:.4f}", flush=True)
        if f1 > best_f1:
            best_f1 = f1
            best_epoch = epoch
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }

    temporary_path = checkpoint.with_suffix(".tmp")
    torch.save(best_state, temporary_path)
    temporary_path.replace(checkpoint)
    print(f"  Best epoch: {best_epoch}. Val macro F1: {best_f1:.4f}", flush=True)


def write_csv(path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


def plot_curve(results, output_dir):
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 5))
    for method, label in (("baseline", "Piso supervisionado"), ("student", "Aluno filtrado")):
        scores = [
            [results[method, budget, seed][0] for seed in seeds]
            for budget, seeds in BUDGET_SEEDS.items()
        ]
        axis.errorbar(
            list(BUDGET_SEEDS), [mean(values) for values in scores],
            yerr=[stdev(values) if len(values) > 1 else 0 for values in scores],
            marker="o", capsize=4, label=label,
        )
    axis.axhline(TEACHER_F1, linestyle="--", color="tab:green", label="Professor BioCLIP")
    axis.axhline(SUPERVISED_REFERENCE, linestyle=":", color="0.35", label="Teto do monitor")
    axis.set_xscale("log")
    axis.set_xticks(
        list(BUDGET_SEEDS), labels=[str(budget) for budget in BUDGET_SEEDS]
    )
    axis.set(xlabel="Rótulos reais por classe", ylabel="F1 macro em test_public", ylim=(0, 1))
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "curva_orcamento.png", dpi=160)
    plt.close(figure)


def main():

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", required=True, type=float, help="Student cutoff selected on val")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--pseudo-labels", type=Path, default=OUTPUT_DIR / "teacher_train.csv")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR / "parte2")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--resume", action="store_true", help="Reuse completed runs with the same inputs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    class_names = load_class_names(args.data_dir / "classes.csv")
    num_classes = len(class_names)
    
    transform = SquareImageTensor(224, imagenet_normalize=True)
    validation = AnimalDataset(root=args.data_dir, split="val", transform=transform)
    validation_loader = DataLoader(validation, batch_size=16, shuffle=False)
    output_dir = args.output_dir / f"threshold_{args.threshold}_epochs_{args.epochs}"
    weights_dir = output_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = {}
    for budget, seeds in BUDGET_SEEDS.items():
        for seed in seeds:
            for method in METHODS:
                key = (method, budget, seed)
                checkpoint = weights_dir / f"{method}_budget_{budget}_seed_{seed}.pt"
                checkpoints[key] = checkpoint
                if args.resume and checkpoint.is_file():
                    print(f"Reusing {checkpoint.name}", flush=True)
                    continue
                train = AnimalDataset(
                    root=args.data_dir, split="train", label_budget=budget, transform=transform
                )
                real_count = len(train)
                if method == "student":
                    retained = add_pseudo_labels(
                        train, args.pseudo_labels, class_names, args.threshold
                    )
                    if not retained:
                        raise ValueError(f"No pseudo-labels retained for budget {budget}")
                print(f"{key}: {real_count} real, {len(train) - real_count} pseudo-labels", flush=True)
                train_checkpoint(
                    train, validation_loader, checkpoint, seed, args.epochs, num_classes, device
                )

    test = AnimalDataset(root=args.data_dir, split="test_public", transform=transform)
    test_loader = DataLoader(test, batch_size=16, shuffle=False)
    model = ResNetClassifier(num_classes=num_classes, pretrained=False).to(device)
    results = {}
    class_scores = {}
    for key, checkpoint in checkpoints.items():
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        confusion, f1 = evaluate_model(model, test_loader, device, num_classes)
        accuracy = confusion.diag().sum().item() / confusion.sum().item()
        results[key] = (f1, accuracy)
        counts = confusion.sum(dim=0) + confusion.sum(dim=1)
        class_scores[key] = (2 * confusion.diag() / counts.clamp_min(1)).tolist()
        print(f"{key}: test_public macro F1={f1:.4f}, accuracy={accuracy:.4f}", flush=True)

    write_csv(
        output_dir / "curva_orcamento.csv",
        ["metodo", "orcamento", "seed", "f1_macro", "acuracia"],
        [(*key, *metrics) for key, metrics in results.items()],
    )
    print("Budget 5, mean ± sample standard deviation across three seeds:")
    for method in METHODS:
        f1_scores, accuracies = zip(
            *(results[method, 5, seed] for seed in BUDGET_SEEDS[5])
        )
        print(
            f"  {method}: F1={mean(f1_scores):.4f} ± {stdev(f1_scores):.4f}, "
            f"accuracy={mean(accuracies):.4f} ± {stdev(accuracies):.4f}"
        )
    rows = []
    for index, label in enumerate(class_names):
        baseline = mean(class_scores["baseline", 5, seed][index] for seed in BUDGET_SEEDS[5])
        student = mean(class_scores["student", 5, seed][index] for seed in BUDGET_SEEDS[5])
        rows.append((label, baseline, student, student - baseline))
    write_csv(
        output_dir / "ganho_por_classe_5.csv",
        ["classe", "baseline_f1_medio", "aluno_f1_medio", "ganho"],
        sorted(rows, key=lambda row: row[-1], reverse=True),
    )
    plot_curve(results, output_dir)
    print(f"Results: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
