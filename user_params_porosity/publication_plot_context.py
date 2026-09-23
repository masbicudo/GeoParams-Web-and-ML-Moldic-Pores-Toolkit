"""Shared Matplotlib settings for publication-ready plot exports."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


PUBLICATION_RC = {
    "font.size": 16,
    "axes.titlesize": 20,
    "axes.labelsize": 21,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "figure.titlesize": 20,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


@contextmanager
def publication_style() -> Iterator[None]:
    """Temporarily apply the font sizing used by the manuscript plots."""

    with plt.rc_context(PUBLICATION_RC):
        yield


def save_pdf(figure: Figure, output_path: str | Path) -> Path:
    """Save a tightly cropped vector PDF and create its parent directory."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_path,
        format="pdf",
        bbox_inches="tight",
        metadata={"Creator": "user_params_porosity"},
    )
    return output_path
