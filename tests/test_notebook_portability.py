import os
import subprocess
import sys
from pathlib import Path

import nbformat


def test_notebook_bootstraps_from_verified_embedded_runtime(project_root: Path, tmp_path: Path):
    notebook_path = project_root / "notebooks/us_treasury_yield_curve_pca.ipynb"
    probe = """
import nbformat
import sys
from pathlib import Path

notebook = nbformat.read(Path(sys.argv[1]), as_version=4)
namespace = {}
exec(compile(notebook.cells[2].source, "<standalone-setup>", "exec"), namespace)
exec(compile(notebook.cells[3].source, "<standalone-binding>", "exec"), namespace)
import yield_curve_pca
assert namespace["RUNTIME_MODE"] == "standalone embedded"
assert namespace["PIPELINE_PROJECT_ROOT"] is None
assert yield_curve_pca.__version__ == "3.3.0"
assert str(namespace["PACKAGE_ROOT"]) in str(Path(yield_curve_pca.__file__).resolve())
print("standalone-bootstrap-pass")
"""
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("YIELD_CURVE_PCA_PROJECT_ROOT", None)
    environment["MPLBACKEND"] = "Agg"
    environment["MPLCONFIGDIR"] = str(tmp_path / "matplotlib")
    environment["IPYTHONDIR"] = str(tmp_path / "ipython")
    completed = subprocess.run(
        [sys.executable, "-c", probe, str(notebook_path)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "standalone-bootstrap-pass" in completed.stdout


def test_notebook_rejects_tampered_embedded_runtime(project_root: Path, tmp_path: Path):
    notebook_path = project_root / "notebooks/us_treasury_yield_curve_pca.ipynb"
    probe = """
import nbformat
import sys
from pathlib import Path

notebook = nbformat.read(Path(sys.argv[1]), as_version=4)
source = notebook.cells[2].source
prefix = "EMBEDDED_RUNTIME_CHUNKS = (\\n    '"
position = source.index(prefix) + len(prefix)
replacement = "A" if source[position] != "A" else "B"
tampered = source[:position] + replacement + source[position + 1:]
try:
    exec(compile(tampered, "<tampered-standalone-setup>", "exec"), {})
except RuntimeError as exc:
    assert "SHA-256" in str(exc)
else:
    raise AssertionError("Tampered embedded runtime was accepted")
print("tamper-rejection-pass")
"""
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("YIELD_CURVE_PCA_PROJECT_ROOT", None)
    environment["MPLBACKEND"] = "Agg"
    environment["MPLCONFIGDIR"] = str(tmp_path / "tamper-matplotlib")
    environment["IPYTHONDIR"] = str(tmp_path / "tamper-ipython")
    completed = subprocess.run(
        [sys.executable, "-c", probe, str(notebook_path)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "tamper-rejection-pass" in completed.stdout


def test_notebook_python_gate_does_not_reject_newer_runtimes(project_root: Path):
    notebook_path = project_root / "notebooks/us_treasury_yield_curve_pca.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    setup_source = notebook.cells[2].source
    assert "if runtime_python_version < (3, 10):" in setup_source
    assert "if runtime_python_version < (3, 14)" in setup_source
    assert "if not (3, 10) <= sys.version_info[:2]" not in setup_source
