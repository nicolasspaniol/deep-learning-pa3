default: typecheck test

test:
    uv run pytest tests

typecheck:
    uvx ty check --exclude "**/*.ipynb"
