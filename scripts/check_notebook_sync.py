"""Check that a committed executed notebook matches its deterministic source generator."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat


def _cell_signature(path: Path) -> list[tuple[str, str, str | None]]:
    notebook = nbformat.read(path, as_version=4)
    return [(cell.cell_type, cell.source, cell.get("id")) for cell in notebook.cells]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("committed", type=Path)
    parser.add_argument("generated", type=Path)
    args = parser.parse_args()
    committed = _cell_signature(args.committed)
    generated = _cell_signature(args.generated)
    if committed != generated:
        differing = [
            index for index, pair in enumerate(zip(committed, generated, strict=False)) if pair[0] != pair[1]
        ]
        raise RuntimeError(
            "Committed notebook sources differ from the generator: "
            f"committed_cells={len(committed)}, generated_cells={len(generated)}, "
            f"differing_zero_based_cells={differing}."
        )
    print(f"Notebook source sync passed: {len(committed)} cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
