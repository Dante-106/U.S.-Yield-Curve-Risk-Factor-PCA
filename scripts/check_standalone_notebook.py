"""Execute a notebook copy with no repository root or project path available."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import nbformat


def _rendered_text(notebook) -> str:
    return "\n".join(
        str(output.get("text", ""))
        + str(output.get("data", {}).get("text/plain", ""))
        + str(output.get("data", {}).get("text/markdown", ""))
        for cell in notebook.cells
        if cell.cell_type == "code"
        for output in cell.get("outputs", [])
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument(
        "--kernel-mode",
        choices=("subprocess", "inprocess"),
        default="subprocess",
    )
    args = parser.parse_args()

    notebook_path = args.notebook.resolve()
    executor = Path(__file__).resolve().with_name("execute_notebook.py")
    with tempfile.TemporaryDirectory(prefix="yield_curve_pca_standalone_check_") as directory:
        isolation_root = Path(directory)
        working_directory = isolation_root / "work"
        working_directory.mkdir()
        isolated_notebook = working_directory / notebook_path.name
        executed_notebook = isolation_root / "executed.ipynb"
        shutil.copyfile(notebook_path, isolated_notebook)
        environment = dict(os.environ)
        environment.pop("PYTHONPATH", None)
        environment.pop("YIELD_CURVE_PCA_PROJECT_ROOT", None)
        environment["MPLBACKEND"] = "Agg"
        environment["MPLCONFIGDIR"] = str(isolation_root / "matplotlib")
        environment["IPYTHONDIR"] = str(isolation_root / "ipython")
        subprocess.run(
            [
                sys.executable,
                str(executor),
                str(isolated_notebook),
                "--output",
                str(executed_notebook),
                "--working-directory",
                str(working_directory),
                "--kernel-mode",
                args.kernel_mode,
                "--timeout",
                str(args.timeout),
            ],
            cwd=working_directory,
            env=environment,
            check=True,
        )
        executed = nbformat.read(executed_notebook, as_version=4)
        rendered_text = _rendered_text(executed)
        required_evidence = (
            "Runtime ready:** standalone embedded",
            "Execution binding verified (standalone embedded)",
            "Pipeline completed:",
            "Figure lifecycle check",
        )
        missing = [evidence for evidence in required_evidence if evidence not in rendered_text]
        if missing:
            raise RuntimeError(f"Standalone notebook evidence is missing: {missing}")
    print(f"Standalone notebook execution passed: {notebook_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
