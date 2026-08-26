"""Execute and validate a notebook without requiring the nbconvert CLI."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import nbformat
from ipykernel.inprocess.blocking import BlockingInProcessKernelClient
from ipykernel.inprocess.manager import InProcessKernelManager
from jupyter_client import KernelManager
from nbclient import NotebookClient


class _NbClientCompatibleInProcessClient(BlockingInProcessKernelClient):
    """Adapt ipykernel's in-process client to nbclient's timeout signature."""

    def wait_for_ready(self, timeout=None):
        return super().wait_for_ready()


class _NbClientCompatibleInProcessManager(InProcessKernelManager):
    """Socket-free kernel manager for locked-down build environments."""

    def client(self, **kwargs):
        options = dict(self.get_connection_info(session=True))
        options.update(
            {
                "connection_file": self.connection_file,
                "parent": self,
                "kernel": self.kernel,
            }
        )
        options.update(kwargs)
        return _NbClientCompatibleInProcessClient(**options)

    def shutdown_kernel(self, now=False):
        return super().shutdown_kernel()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--working-directory", type=Path)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument(
        "--kernel-mode",
        choices=("subprocess", "inprocess"),
        default="subprocess",
        help="Use inprocess only when the execution sandbox prohibits local sockets.",
    )
    args = parser.parse_args()

    notebook_path = args.notebook.resolve()
    working_directory = (
        args.working_directory.resolve() if args.working_directory is not None else notebook_path.parent
    )
    if not working_directory.is_dir():
        raise NotADirectoryError(f"Notebook working directory does not exist: {working_directory}")
    notebook = nbformat.read(notebook_path, as_version=4)
    manager: KernelManager | InProcessKernelManager | None = None
    if args.kernel_mode == "inprocess":
        manager = _NbClientCompatibleInProcessManager(kernel_name="python3")

    client = NotebookClient(
        notebook,
        km=manager,
        timeout=args.timeout,
        kernel_name="python3",
        allow_errors=False,
        resources={"metadata": {"path": str(working_directory)}},
    )
    executed = client.execute()
    nbformat.validate(executed)

    code_cells = [cell for cell in executed.cells if cell.cell_type == "code"]
    if any(cell.execution_count is None for cell in code_cells):
        raise RuntimeError("At least one code cell lacks an execution count.")
    errors = [
        output
        for cell in code_cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    if errors:
        raise RuntimeError(f"Notebook contains {len(errors)} error outputs.")

    if args.output is not None:
        output_path = args.output.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".ipynb", dir=output_path.parent, delete=False, encoding="utf-8"
        ) as handle:
            temporary = Path(handle.name)
            nbformat.write(executed, handle)
        temporary.replace(output_path)
    print(f"Executed {len(code_cells)} code cells successfully: {notebook_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
