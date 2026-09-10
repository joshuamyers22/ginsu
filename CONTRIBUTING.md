# Contribution guidelines

## What to work on?

You're welcome to propose and contribute new ideas.
Open a discussion in the Ginsu repository before starting a potentially
out-of-scope or compatibility-sensitive change:
https://github.com/joshuamyers22/ginsu/discussions.
It's generally a good idea to have a quick discussion before opening a pull request that is potentially out-of-scope.

## Fork/clone/pull

The typical workflow for contributing to `ginsu` is:

1. Fork the Ginsu repository's `main` branch:
   https://github.com/joshuamyers22/ginsu.
2. Clone your fork locally.
3. Commit changes.
4. Push the changes to your fork.
5. Send a pull request from your fork back to the original `main` branch.

## Local setup

We encourage you to use a virtual environment. You'll want to activate it every time you work on `ginsu`.

Install dependencies via uv:

```sh
$ make init
```

Or manually:

```sh
$ pip install uv
$ uv sync --all-extras
```

Install the [pre-commit](https://pre-commit.com/) push hooks. This will run some code quality checks every time you push to GitHub.

```sh
$ pre-commit install --hook-type pre-push
```

You can optionally run `pre-commit` at any time as so:

```sh
$ pre-commit run --all-files
```

## Code quality

We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting. Run the following to check and fix issues:

```sh
$ make lint
```

Or to only check without fixing:

```sh
$ make check
```

## Documenting your change

If you're adding a class or a function, then you'll need to add a docstring. We follow the [numpydoc docstring convention](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html), so please do too.

In order to build the documentation locally, run:

```sh
$ make doc
```

Run executable documentation examples with `make doctest`. Both the HTML build
and documentation examples are included in `make check`.

## Adding a release note

Add user-visible changes to the `Unreleased` section of
[`CHANGELOG.md`](CHANGELOG.md). New public classes and functions must also be
added to the appropriate API or task-oriented documentation page.

## Testing

**Unit tests**

These tests absolutely have to pass.

```sh
$ make test
```

Or directly with pytest:

```sh
$ uv run pytest tests/
```

**Notebook tests**

The maintained notebooks are deterministic, offline, and part of the push
gate. They execute in a dedicated environment without pandas. Generated
notebooks are written under the ignored `docs/build/notebooks` directory;
the checked-in sources keep outputs cleared.

```sh
$ make execute-notebooks
```
