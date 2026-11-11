from collections.abc import Sequence


def build_prompts(class_names: Sequence[str]) -> list[str]:
    """Build one prompt per class, preserving the order of the class names."""
    return [
        f"a camera trap photo of the {name.replace('_', ' ')}."
        for name in class_names
    ]


def main() -> None:
    from deep_learning_pa3.evaluation import load_class_names
    from deep_learning_pa3.paths import CLASSES_PATH

    try:
        class_names = load_class_names(CLASSES_PATH)
    except FileNotFoundError:
        raise SystemExit(f"Extract the PA3 classes.csv to {CLASSES_PATH}") from None

    for label, prompt in zip(class_names, build_prompts(class_names), strict=True):
        print(f"{label}\t{prompt}")


if __name__ == "__main__":
    main()