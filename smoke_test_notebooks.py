"""Smoke-test notebook path hygiene and required local data layout.

This is intentionally lightweight: it does not execute long-running training
notebooks. Instead, it catches the most common public-repo failure mode for
these notebooks: stale paths inherited from the historical research workspace.
"""

from __future__ import annotations

import json
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parent

NOTEBOOK_DIRS = [
    REPO_DIR / "geo_params_web" / "notebooks",
    REPO_DIR / "ml_moldic_pores" / "notebooks",
]

STALE_PATTERNS = [
    "../data/classificada_01",
    "../data/thin_sections",
    "../data/c_min_k_max_params.csv",
    "../out/",
    "../models/",
    "classificada_01",
]

REQUIRED_PATHS = [
    REPO_DIR / "datasets" / "article_thin_sections",
    REPO_DIR / "datasets" / "pore_type_training",
    REPO_DIR / "geo_params_web" / "static" / "imgs_sections" / "metadata.json",
    REPO_DIR / "geo_params_web" / "prepare_web_images.py",
    REPO_DIR / "ml_moldic_pores" / "data" / "c_min_k_max_params.csv",
    REPO_DIR / "ml_moldic_pores" / "out" / "pore_type_training" / "ML-tste_original_25.jpg",
]


def notebook_source_text(notebook_file: Path) -> str:
    with notebook_file.open("r", encoding="utf-8") as fp:
        notebook = json.load(fp)

    source_chunks: list[str] = []
    for cell in notebook.get("cells", []):
        source = cell.get("source", [])
        if isinstance(source, list):
            source_chunks.append("".join(source))
        elif isinstance(source, str):
            source_chunks.append(source)
    return "\n".join(source_chunks)


def main() -> int:
    failures: list[str] = []

    for notebook_dir in NOTEBOOK_DIRS:
        for notebook_file in sorted(notebook_dir.glob("*.ipynb")):
            text = notebook_source_text(notebook_file)
            for pattern in STALE_PATTERNS:
                if pattern in text:
                    failures.append(
                        f"{notebook_file.relative_to(REPO_DIR)} contains stale path: {pattern}"
                    )

    for required_path in REQUIRED_PATHS:
        if not required_path.exists():
            failures.append(
                f"Required path is missing: {required_path.relative_to(REPO_DIR)}"
            )

    if failures:
        print("Notebook smoke test failed:")
        for failure in failures:
            print(f"  - {failure}")
        print()
        print("Run the setup commands from README.md, especially:")
        print("  cd geo_params_web")
        print("  pdm run python prepare_web_images.py")
        print("  cd ../ml_moldic_pores")
        print("  pdm run python imports.py")
        return 1

    print("Notebook smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
