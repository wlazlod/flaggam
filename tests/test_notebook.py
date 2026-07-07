"""Offline check that the guidebook notebook is valid and was executed with outputs."""

import json
from pathlib import Path

NOTEBOOK = Path(__file__).parent.parent / "docs" / "notebooks" / "german_credit.ipynb"


def test_german_credit_notebook_has_executed_outputs() -> None:
    nb = json.loads(NOTEBOOK.read_text())
    assert nb["nbformat"] >= 4

    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(code_cells) > 0

    cells_with_outputs = [c for c in code_cells if c.get("outputs")]
    assert len(cells_with_outputs) >= 1
