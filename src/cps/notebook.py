from __future__ import annotations

import json
import platform
import shutil
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable

from .plot_labels import compact_basis_labels


def _display_markdown(text: str) -> None:
    try:
        from IPython.display import Markdown, display

        display(Markdown(text))
    except ImportError:  # pragma: no cover - notebook convenience
        print(text)


def lesson(title: str, body: str) -> None:
    """Render a visible pedagogical waypoint in a notebook."""

    _display_markdown(f"## {title}\n\n{body}")


def evidence_boundary(*items: str) -> None:
    rendered = "\n".join(f"- {item}" for item in items)
    _display_markdown(f"### Evidence boundary\n\n{rendered}")


def show_environment() -> dict[str, Any]:
    import numpy as np

    payload: dict[str, Any] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    try:
        import torch

        payload.update(
            {
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
                "cuda_version": torch.version.cuda,
            }
        )
    except ImportError:
        payload["torch"] = "not installed"
    try:
        import transformers

        payload["transformers"] = transformers.__version__
    except ImportError:
        payload["transformers"] = "not installed"
    _display_markdown(
        "### Runtime inventory\n\n"
        + "\n".join(f"- **{key.replace('_', ' ')}:** `{value}`" for key, value in payload.items())
    )
    return payload


def show_config(config: Any) -> dict[str, Any]:
    if hasattr(config, "to_dict"):
        payload = config.to_dict()
    elif is_dataclass(config):
        payload = asdict(config)
    elif isinstance(config, dict):
        payload = config
    else:
        raise TypeError("config must be a dataclass, mapping, or expose to_dict()")
    _display_markdown(
        "### Resolved experiment contract\n\n"
        "The configuration below is the contract actually passed to the runner. "
        "In particular, note the attention implementation and the permitted JVP fallback.\n\n"
        f"```json\n{json.dumps(payload, indent=2)}\n```"
    )
    return payload


def display_probe_summary(root: str | Path, *, top_n: int = 8) -> dict[str, Any]:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    records = json.loads((root / "couplings.json").read_text(encoding="utf-8"))
    basis = json.loads((root / "basis.json").read_text(encoding="utf-8"))
    matrix = np.load(root / "reduced_operator.npy")
    labels, full_to_short, shared_block, label_key = compact_basis_labels(basis)

    jacobian = manifest.get("jacobian", {})
    fallback = jacobian.get("fallback_used", False)
    backend_note = (
        "The exact forward-mode path was unavailable, so this packet uses the declared "
        "centered finite-difference fallback."
        if fallback
        else "The projected columns were produced by the requested autodiff backend."
    )
    _display_markdown(
        "## Read the evidence packet\n\n"
        f"- **Output:** `{root}`\n"
        f"- **Projected rank:** `{manifest['basis_rank']}`\n"
        f"- **JVP backend:** `{jacobian.get('effective_backend', 'legacy/unknown')}`\n"
        f"- **Attention backend:** `{manifest.get('attention', {}).get('active_implementation', 'unknown')}`\n"
        f"- **Maximum closure residual:** `{manifest['projection']['maximum_closure_residual']:.6g}`\n\n"
        f"{backend_note}"
    )

    key_lines = "\n".join(f"- `{short}` — `{full}`" for short, full in label_key)
    block_note = (
        f"All displayed modes belong to the compact block **`{shared_block}`**. "
        if shared_block is not None
        else "Labels retain compact block identity because several parameter blocks are present. "
    )
    _display_markdown(
        "### Basis-label key\n\n"
        + block_note
        + "The evidence packet retains the exact basis names.\n\n"
        + key_lines
    )

    rows = []
    for record in records:
        metrics = record["metrics"]
        rows.append(
            {
                "source": full_to_short.get(record["source"], record["source"]),
                "target": full_to_short.get(record["target"], record["target"]),
                "|coupling|": record["magnitude"],
                "max spectral radius": metrics["spectral_radius_max"],
                "finite-horizon gain": metrics["finite_horizon_gain"],
                "minimum gap": metrics["minimum_gap"],
                "max eigenvalue condition": metrics["maximum_eigenvalue_condition"],
                "loop length": metrics["total_loop_length"],
            }
        )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(
            ["max spectral radius", "finite-horizon gain"], ascending=False
        ).reset_index(drop=True)
        try:
            from IPython.display import display

            display(frame.head(top_n))
        except ImportError:  # pragma: no cover
            print(frame.head(top_n).to_string(index=False))

    figure_width = max(6.5, 1.15 * len(labels) + 2.0)
    figure_height = max(4.8, 0.9 * len(labels) + 1.5)
    figure, axis = plt.subplots(
        figsize=(figure_width, figure_height),
        constrained_layout=True,
    )
    image = axis.imshow(np.abs(matrix))
    title = "Magnitude of the projected optimizer-state Jacobian"
    if shared_block is not None:
        title += f"\nBlock: {shared_block}"
    axis.set_title(title)
    axis.set_xlabel("source basis mode")
    axis.set_ylabel("target basis mode")
    rotation = 0 if max((len(label) for label in labels), default=0) <= 14 else 30
    horizontal_alignment = "center" if rotation == 0 else "right"
    axis.set_xticks(
        range(len(labels)),
        labels,
        rotation=rotation,
        ha=horizontal_alignment,
    )
    axis.set_yticks(range(len(labels)), labels)
    axis.tick_params(axis="both", labelsize=9)
    figure.colorbar(image, ax=axis, label="|Âᵢⱼ|")
    plt.show()

    if not frame.empty:
        plot_frame = frame.head(top_n).copy()
        plot_frame["coupling"] = plot_frame["source"] + " → " + plot_frame["target"]
        axis = plot_frame.plot.barh(
            x="coupling",
            y="max spectral radius",
            legend=False,
            figsize=(8.5, max(4, 0.55 * len(plot_frame))),
        )
        axis.set_title("Worst phase-envelope spectral radius by coupling")
        axis.set_xlabel("maxφ ρ(Aφ)")
        axis.set_ylabel("directed basis coupling")
        axis.invert_yaxis()
        plt.tight_layout()
        plt.show()

    _display_markdown(
        "### How to read these diagnostics\n\n"
        "1. **Closure residual** asks whether the chosen semantic subspace captures the image of each probe direction. A large value is a warning about the projection, not automatically about the optimizer.\n"
        "2. **Maximum spectral radius** asks how close a phase-rotated coupling can bring the reduced map to expansion.\n"
        "3. **Finite-horizon gain** detects transient amplification that eigenvalues alone can miss.\n"
        "4. **Minimum gap and eigenvalue conditioning** flag mode collisions and fragile eigenvectors.\n"
        "5. Compact labels are presentation aliases only; the manifest and basis registry retain the exact parameter paths.\n"
        "6. These are local, projected diagnostics. They become optimizer evidence only after prospective prediction and matched continuation tests."
    )
    return {
        "manifest": manifest,
        "records": records,
        "basis": basis,
        "matrix": matrix,
        "table": frame,
        "label_key": label_key,
    }


def display_longitudinal_summary(summary_path: str | Path) -> Any:
    import pandas as pd

    summary_path = Path(summary_path)
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = []
    for root_text in payload["roots"]:
        root = Path(root_text)
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        records = json.loads((root / "couplings.json").read_text(encoding="utf-8"))
        rows.append(
            {
                "revision": manifest["revision"],
                "maximum closure residual": manifest["projection"]["maximum_closure_residual"],
                "JVP backend": manifest.get("jacobian", {}).get("effective_backend", "legacy/unknown"),
                "max phase spectral radius": max(
                    (item["metrics"]["spectral_radius_max"] for item in records), default=float("nan")
                ),
                "max finite-horizon gain": max(
                    (item["metrics"]["finite_horizon_gain"] for item in records), default=float("nan")
                ),
            }
        )
    frame = pd.DataFrame(rows)
    try:
        from IPython.display import display

        display(frame)
    except ImportError:  # pragma: no cover
        print(frame.to_string(index=False))
    return frame


def export_artifacts(
    sources: Iterable[str | Path] = ("/content/cps-artifacts",),
    *,
    export_root: str | Path = "/content/cps-export",
) -> Path:
    export_root = Path(export_root)
    export_root.mkdir(parents=True, exist_ok=True)
    copied = 0
    for source_text in sources:
        source = Path(source_text)
        if not source.exists():
            continue
        target = export_root / source.name
        if source.is_dir():
            shutil.copytree(source, target, dirs_exist_ok=True)
        else:
            shutil.copy2(source, target)
        copied += 1
    archive = Path(shutil.make_archive(str(export_root), "zip", export_root))
    _display_markdown(
        "## Export complete\n\n"
        f"Copied **{copied}** evidence source(s). The Colab CLI can retrieve `{archive}` "
        "after notebook execution."
    )
    return archive
