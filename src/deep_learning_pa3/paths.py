"""Shared paths for repository data and generated artifacts."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
CLASSES_PATH = DATA_DIR / "classes.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
WEIGHTS_DIR = PROJECT_ROOT / "weights"

CNN_WEIGHTS_PATH = WEIGHTS_DIR / "cnn_classifier.pt"
RESNET_WEIGHTS_PATH = WEIGHTS_DIR / "resnet_classifier.pt"
