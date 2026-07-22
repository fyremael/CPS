from __future__ import annotations

import ast
from pathlib import Path

import nbformat


def main() -> None:
    failures: list[str] = []
    for path in sorted(Path("notebooks").glob("*.ipynb")):
        notebook = nbformat.read(path, as_version=4)
        if notebook.nbformat != 4:
            failures.append(f"{path}: expected nbformat 4")
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type == "code":
                try:
                    ast.parse(cell.source)
                except SyntaxError as exc:
                    failures.append(f"{path}: cell {index}: {exc}")
            if cell.get("outputs"):
                failures.append(f"{path}: cell {index}: committed outputs are prohibited")
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"validated {len(list(Path('notebooks').glob('*.ipynb')))} notebooks")


if __name__ == "__main__":
    main()
