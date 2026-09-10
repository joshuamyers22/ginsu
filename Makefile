init:
	python3 -m pip install --upgrade pip
	pip3 install uv
	uv sync --frozen --all-extras

lint:
	uv run ruff check . --fix
	uv run ruff format .

check:
	uv run --frozen ruff check .
	uv run --frozen ruff format --check .
	uv run --frozen mypy
	uv run --frozen pytest
	LC_ALL=C LANG=C uv run --frozen sphinx-build -W -a -E docs/source docs/build

typecheck:
	uv run --frozen mypy

test:
	uv run --frozen coverage run -m pytest
	uv run --frozen coverage report -m

benchmark:
	uv run --frozen pytest -m performance

doc:
	LC_ALL=C LANG=C uv run --frozen sphinx-build -W -a -E docs/source docs/build

notebook:
	uv run --frozen --extra notebooks --extra plot jupyter notebook

execute-notebooks:
	uv run --frozen --extra notebooks --extra plot jupyter nbconvert --execute --to notebook --output-dir docs/build/notebooks notebooks/*.ipynb --ExecutePreprocessor.timeout=600
