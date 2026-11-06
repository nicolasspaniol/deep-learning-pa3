default: typecheck test

test:
    uv run pytest

typecheck:
    uvx ty check --exclude "**/*.ipynb"
