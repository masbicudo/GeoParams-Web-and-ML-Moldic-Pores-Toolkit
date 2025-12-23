"""
Quick Test — Environment and Data Check

This script provides a minimal, non-interactive test to verify that:
- the Python environment is correctly configured
- required dependencies are available
- the processed user-parameterization dataset can be loaded
- the expected data schema is present

It does not require the full image dataset or GPU acceleration.
"""

from pathlib import Path
import sys

import pandas as pd


def fail(msg: str) -> None:
    print(f"[FAIL] {msg}")
    sys.exit(1)


def ok(msg: str) -> None:
    print(f"[OK] {msg}")


# ---------------------------------------------------------------------
# Locate dataset
# ---------------------------------------------------------------------

csv_path = Path(__file__).resolve().parent / "static/output/clicks_data.csv"

if not csv_path.exists():
    fail(
        "clicks_data.csv was not found.\n"
        "Expected location:\n"
        f"  {csv_path}\n"
        "Please ensure the repository was cloned correctly."
    )

ok("Found clicks_data.csv")


# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------

try:
    df = pd.read_csv(csv_path)
except Exception as e:
    fail(f"Failed to read clicks_data.csv: {e}")

ok(f"Loaded dataset with {len(df)} rows")


# ---------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------

required_columns = {
    "source_image_key",
    "x",
    "y",
    "sz_kind",
    "cut_kind",
    "porosity",
    "pore_count",
    "experience",
}

missing = required_columns - set(df.columns)
if missing:
    fail(f"Missing expected columns: {sorted(missing)}")

ok("All required columns are present")


# ---------------------------------------------------------------------
# Basic sanity checks
# ---------------------------------------------------------------------

if df.empty:
    fail("Dataset is empty")

if df["experience"].isnull().any():
    fail("Found null values in 'experience' column")

ok("Basic sanity checks passed")


# ---------------------------------------------------------------------
# Simple summaries (printed, no plots)
# ---------------------------------------------------------------------

print("\nPorosity summary by experience level:")
print(df.groupby("experience")["porosity"].describe())

print("\nMean porosity by cut type:")
print(df.groupby("cut_kind")["porosity"].mean())

print("\nQuick test completed successfully.")
print("Environment and core data structures are correctly configured.")
