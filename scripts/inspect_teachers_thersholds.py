import argparse
import csv
from collections import Counter
from math import ceil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--pseudo-labels", type=Path, default=ROOT / "outputs/teacher_train.csv")
parser.add_argument("--real-labels", type=Path, default=ROOT / "data/labels/rotulos_5.csv")
args = parser.parse_args()


with args.real_labels.open(newline="", encoding="utf-8") as file:
    real_ids = {row["image_id"] for row in csv.DictReader(file)}
with args.pseudo_labels.open(newline="", encoding="utf-8") as file:
    predictions = [
        (row["label"], float(row["confidence"]))
        for row in csv.DictReader(file)
        if row["image_id"] not in real_ids
    ]

scores = sorted((score for _, score in predictions), reverse=True)

thresholds = {0.0, 0.8}
for fraction in (0.75, 0.50, 0.25):
    thresholds.add(scores[ceil(len(scores) * fraction) - 1])

total_by_class = Counter(label for label, _ in predictions)
for threshold in sorted(thresholds):
    kept = Counter(label for label, score in predictions if score >= threshold)
    count = kept.total()
    print(f"threshold={threshold}  retained={count}/{len(scores)} ({count / len(scores):.1%})")
    