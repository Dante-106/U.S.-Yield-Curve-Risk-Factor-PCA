import io
import tokenize
from pathlib import Path

import nbformat


def _python_comments(source: str) -> list[tuple[int, str]]:
    return [
        (token.start[0], token.string)
        for token in tokenize.generate_tokens(io.StringIO(source).readline)
        if token.type == tokenize.COMMENT
    ]


def test_python_sources_contain_no_inline_comments(project_root: Path):
    source_paths = sorted(
        [
            *project_root.joinpath("src").rglob("*.py"),
            *project_root.joinpath("scripts").rglob("*.py"),
            *project_root.joinpath("tests").rglob("*.py"),
        ]
    )
    findings = {
        path.relative_to(project_root).as_posix(): _python_comments(path.read_text(encoding="utf-8"))
        for path in source_paths
    }
    assert not {path: comments for path, comments in findings.items() if comments}


def test_notebook_code_cells_contain_no_inline_comments(project_root: Path):
    notebook_path = project_root / "notebooks/us_treasury_yield_curve_pca.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    findings = {
        index: _python_comments(cell.source)
        for index, cell in enumerate(notebook.cells)
        if cell.cell_type == "code"
    }
    assert not {index: comments for index, comments in findings.items() if comments}
