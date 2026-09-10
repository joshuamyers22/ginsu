"""Static contracts for deterministic, pandas-free maintained notebooks."""

from __future__ import annotations

import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_NOTEBOOKS = (
    _ROOT / "notebooks/1. Implementing Ginsu on Titanic dataset.ipynb",
    _ROOT
    / "notebooks/2. Implementing Ginsu on California housing dataset.ipynb",
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _source(notebook: dict) -> str:
    return "\n".join("".join(cell["source"]) for cell in notebook["cells"])


def test_maintained_notebooks_are_clean_offline_polars_documents() -> None:
    for path in _NOTEBOOKS:
        notebook = _load(path)
        assert notebook["nbformat"] == 4
        cell_ids = [cell["id"] for cell in notebook["cells"]]
        assert len(cell_ids) == len(set(cell_ids))
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                assert cell["execution_count"] is None
                assert cell["outputs"] == []

        source = _source(notebook)
        assert "import polars as pl" in source
        assert "DiscretizationPlan" in source
        assert "validate_slices" in source
        assert "plot_search_report" in source
        assert "synthetic" in source.lower()
        assert "import pandas" not in source
        assert "import pyarrow" not in source
        assert "import requests" not in source
        assert "fetch_openml" not in source
        assert "fetch_california_housing" not in source
        assert "urlopen(" not in source
        assert ".query(" not in source
        assert "from sliceline" not in source


def test_notebooks_cover_selection_comparison_and_stability_journeys() -> None:
    classification = _source(_load(_NOTEBOOKS[0]))
    regression = _source(_load(_NOTEBOOKS[1]))

    assert "select_slices" in classification
    assert "membership_frame" in classification
    assert "plot_error_dependence" in classification
    assert "SliceAnalysis.from_finder" in regression
    assert "compare_analyses" in regression
    assert "evaluate_stability" in regression
