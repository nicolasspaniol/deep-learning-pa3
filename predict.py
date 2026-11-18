"""Classify a folder of images with the PA3 student on CPU, using local weights."""

import argparse
import csv
import os
from pathlib import Path
import sys
import tempfile

import torch
from PIL import Image


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from deep_learning_pa3.resnet_classifier import ResNetClassifier
from deep_learning_pa3.utils import SquareImageTensor, count_parameters


DEFAULT_WEIGHTS = ROOT / "weights" / "student_threshold_0.9445663690567017_seed_42.pt"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imagens", type=Path, required=True, help="Pasta com as imagens")
    parser.add_argument("--saida", type=Path, required=True, help="CSV de destino")
    parser.add_argument("--pesos", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--classes", type=Path, default=ROOT / "data" / "classes.csv")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    if not args.imagens.is_dir():
        parser.error(f"Pasta de imagens inexistente: {args.imagens}")
    for path in (args.pesos, args.classes):
        if not path.is_file():
            parser.error(f"Arquivo inexistente: {path}")
    if args.batch_size < 1:
        parser.error("--batch-size deve ser positivo")

    with args.classes.open(newline="", encoding="utf-8-sig") as file:
        rows = sorted(csv.DictReader(file), key=lambda row: int(row["idx"]))
    class_names = [row["label"] for row in rows]
    if not rows or [int(row["idx"]) for row in rows] != list(range(len(rows))):
        parser.error("classes.csv deve ter índices contíguos começando em zero")
    if len(set(class_names)) != len(class_names) or any(not name for name in class_names):
        parser.error("classes.csv deve ter nomes únicos e preenchidos")

    image_paths = sorted(
        path for path in args.imagens.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        parser.error(f"Nenhuma imagem encontrada em {args.imagens}")
    if len({path.stem for path in image_paths}) != len(image_paths):
        parser.error("Há imagens com o mesmo image_id (nome sem extensão)")
    protected_paths = [args.pesos, args.classes, *image_paths]
    if args.saida.resolve() in {path.resolve() for path in protected_paths}:
        parser.error("--saida deve apontar para um arquivo diferente dos dados e pesos")

    torch.set_num_threads(min(4, torch.get_num_threads()))
    model = ResNetClassifier(num_classes=len(class_names), pretrained=False).cpu()
    state = torch.load(args.pesos, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    parameters = count_parameters(model)
    print(f"Parâmetros do modelo: {parameters:,}")
    if parameters > 12_000_000:
        parser.error("O modelo ultrapassa o limite de 12 milhões de parâmetros")
    transform = SquareImageTensor(224, imagenet_normalize=True)

    args.saida.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", newline="", encoding="utf-8", delete=False,
            dir=args.saida.parent, prefix=f".{args.saida.name}.", suffix=".tmp",
        ) as file:
            temporary_path = Path(file.name)
            writer = csv.writer(file)
            writer.writerow(["image_id", "label"])
            with torch.inference_mode():
                for start in range(0, len(image_paths), args.batch_size):
                    batch_paths = image_paths[start:start + args.batch_size]
                    tensors = []
                    for path in batch_paths:
                        with Image.open(path) as image:
                            tensors.append(transform(image.convert("RGB")))
                    predictions = model(torch.stack(tensors)).argmax(dim=1).tolist()
                    writer.writerows(
                        (path.stem, class_names[index])
                        for path, index in zip(batch_paths, predictions, strict=True)
                    )
        os.replace(temporary_path, args.saida)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    print(f"CPU: {len(image_paths)} imagens. Predições salvas em {args.saida}")


if __name__ == "__main__":
    main()
