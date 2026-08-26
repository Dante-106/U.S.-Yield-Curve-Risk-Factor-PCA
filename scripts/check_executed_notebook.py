"""Fail when the committed reader notebook is stale, unexecuted, or contains errors."""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat


def main(path: Path) -> int:
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    if not code_cells:
        raise ValueError("Notebook has no code cells.")
    execution_counts = [cell.execution_count for cell in code_cells]
    if any(not isinstance(count, int) for count in execution_counts):
        raise ValueError("Every committed code cell must have an execution count.")
    if execution_counts != list(range(1, len(code_cells) + 1)):
        raise ValueError("Committed notebook code cells must be executed once, in order.")
    errors = [output for cell in code_cells for output in cell.outputs if output.output_type == "error"]
    if errors:
        raise ValueError(f"Committed notebook contains {len(errors)} error outputs.")
    if not any(cell.outputs for cell in code_cells):
        raise ValueError("Committed notebook contains no outputs.")
    output_text = "\n".join(
        str(output.get("text", ""))
        + str(output.get("data", {}).get("text/plain", ""))
        + str(output.get("data", {}).get("text/markdown", ""))
        for cell in code_cells
        for output in cell.outputs
    )
    if "Figure lifecycle check" not in output_text or "[]" not in output_text:
        raise ValueError("Notebook does not prove that matplotlib figures were closed.")
    required_portability_evidence = (
        "Runtime ready:** standalone embedded",
        "Execution binding verified (standalone embedded)",
    )
    missing_portability_evidence = [
        evidence for evidence in required_portability_evidence if evidence not in output_text
    ]
    if missing_portability_evidence:
        raise ValueError(
            f"Committed notebook lacks standalone execution evidence: {missing_portability_evidence}"
        )
    image_count = sum(
        int("image/png" in output.get("data", {}))
        for cell in code_cells
        for output in cell.outputs
    )
    if image_count != 5:
        raise ValueError(f"Committed notebook must contain exactly five PNG figures; found {image_count}.")
    print(
        f"validated executed notebook: {len(code_cells)} code cells, "
        f"{sum(len(cell.outputs) for cell in code_cells)} outputs, {image_count} PNG figures, zero errors"
    )
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: check_executed_notebook.py NOTEBOOK")
    raise SystemExit(main(Path(sys.argv[1])))
