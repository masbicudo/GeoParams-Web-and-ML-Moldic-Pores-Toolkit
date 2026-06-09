"""
Import input data needed by the user-parameter porosity subproject.
"""

from __future__ import annotations

import shutil
from pathlib import Path


DATA_DIR = Path("data")
PARAMS_TARGET = DATA_DIR / "c_min_k_max_params.csv"

PARAMS_CANDIDATES = [
    Path("../geo_params_web/exports/c_min_k_max_params.csv"),
    Path("../ml_moldic_pores/data/c_min_k_max_params.csv"),
]


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for candidate in PARAMS_CANDIDATES:
        if candidate.exists():
            shutil.copy(candidate, PARAMS_TARGET)
            print(f"Copied {candidate} to {PARAMS_TARGET}")
            return 0

    print("Warning: c_min_k_max_params.csv was not found.")
    print("Run this first:")
    print("  cd ../geo_params_web")
    print("  pdm run python exports.py")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
