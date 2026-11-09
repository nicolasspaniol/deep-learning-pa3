"""Report the CNN's macro F1 and confusion matrix on test_public.

This is report-only evaluation. Do not use test_public to select or tune models.
"""

import torch
from torch.utils.data import DataLoader

from deep_learning_pa3.cnn_classifier import CNNClassifier
from deep_learning_pa3.dataset import AnimalDataset
from deep_learning_pa3.evaluation import evaluate_model, load_class_names, save_confusion_report
from deep_learning_pa3.paths import (
    CLASSES_PATH,
    CNN_WEIGHTS_PATH,
    DATA_DIR,
    OUTPUT_DIR,
)
from deep_learning_pa3.utils import SquareImageTensor


IMAGE_SIZE = 128
BATCH_SIZE = 16
NUM_CLASSES = 21


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = AnimalDataset(
        root=DATA_DIR,
        split="test_public",
        transform=SquareImageTensor(IMAGE_SIZE),
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = CNNClassifier(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(CNN_WEIGHTS_PATH, map_location=device, weights_only=True))
    confusion, macro_f1 = evaluate_model(model, loader, device, NUM_CLASSES)
    matrix_path, plot_path = save_confusion_report(
        confusion,
        load_class_names(CLASSES_PATH),
        OUTPUT_DIR,
        "cnn",
        macro_f1,
    )

    print(f"test_public macro F1: {macro_f1:.4f}")
    print("Confusion matrix (rows=true, columns=predicted):")
    print(confusion)
    print(f"Saved labeled confusion matrix to {matrix_path}")
    print(f"Saved confusion matrix plot to {plot_path}")


if __name__ == "__main__":
    main()
