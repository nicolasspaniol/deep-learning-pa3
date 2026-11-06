default: typecheck test

test:
    uv run pytest

typecheck:
    uvx ty check --exclude "**/*.ipynb"

show-dataset:
    uv run python scripts/test_dataset.py
