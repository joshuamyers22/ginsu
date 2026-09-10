"""Static package-boundary regression tests."""

import ast
from pathlib import Path


def test_production_package_has_no_pandas_or_sliceline_imports():
    package_root = Path(__file__).parents[1] / "ginsu"
    forbidden = []

    for path in package_root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported = [
                    alias.name.split(".", maxsplit=1)[0]
                    for alias in node.names
                ]
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported = [node.module.split(".", maxsplit=1)[0]]
            else:
                continue
            for name in imported:
                if name in {"pandas", "sliceline"}:
                    forbidden.append(f"{path.name}:{node.lineno}:{name}")

    assert forbidden == []
