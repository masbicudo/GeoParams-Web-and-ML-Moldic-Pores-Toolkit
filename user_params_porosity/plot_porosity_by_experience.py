"""Generate porosity-by-experience boxplots from individual estimates."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import textwrap
import unicodedata

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from publication_plot_context import publication_style, save_pdf


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "plots" / "outputs"
EXPERIENCE_LEVELS = (1, 2, 3, 4, 5)
REQUIRED_COLUMNS = {"image", "scale", "scale_factor", "experience", "porosity"}


@dataclass(frozen=True)
class ImageSet:
    key: str
    title: str
    results_path: Path


DEFAULT_IMAGE_SETS = (
    ImageSet(
        key="article",
        title="Original article thin sections",
        results_path=(
            PROJECT_DIR
            / "data"
            / "output"
            / "article_thin_sections"
            / "porosity_by_parameter.csv"
        ),
    ),
    ImageSet(
        key="revision",
        title="Final revision thin sections",
        results_path=(
            PROJECT_DIR
            / "data"
            / "output"
            / "generalization_test_thin_sections"
            / "porosity_by_parameter.csv"
        ),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate fixed-scale PDF boxplots of individual porosity estimates "
            "by user experience level."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="PDF output directory. Default: plots/outputs.",
    )
    parser.add_argument(
        "--article-results",
        type=Path,
        default=DEFAULT_IMAGE_SETS[0].results_path,
        help="Article porosity_by_parameter.csv path.",
    )
    parser.add_argument(
        "--revision-results",
        type=Path,
        default=DEFAULT_IMAGE_SETS[1].results_path,
        help="Revision porosity_by_parameter.csv path.",
    )
    return parser.parse_args()


def load_estimates(results_path: Path) -> pd.DataFrame:
    if not results_path.exists():
        raise FileNotFoundError(
            f"Missing porosity estimates: {results_path}\n"
            "Run the corresponding default analysis first."
        )

    frame = pd.read_csv(results_path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns in {results_path}: {sorted(missing)}")

    frame = frame.copy()
    for column in ("scale_factor", "experience", "porosity"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")

    if frame.empty:
        raise ValueError(f"No porosity estimates found in {results_path}.")
    if not frame["experience"].isin(EXPERIENCE_LEVELS).all():
        invalid = sorted(frame.loc[~frame["experience"].isin(EXPERIENCE_LEVELS), "experience"].unique())
        raise ValueError(f"Unexpected experience levels in {results_path}: {invalid}")
    if not frame["porosity"].between(0.0, 1.0).all():
        raise ValueError(f"Porosity values outside [0, 1] found in {results_path}.")

    frame["experience"] = frame["experience"].astype(int)
    return frame


def select_highest_resolution(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep the largest available scale factor independently for each image."""

    highest = frame.groupby("image")["scale_factor"].transform("max")
    selected = frame[np.isclose(frame["scale_factor"], highest)].copy()
    if selected.empty:
        raise ValueError("No rows remained after selecting the highest resolution.")
    return selected


def filename_slug(value: str) -> str:
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_value.lower()).strip("-")
    return slug or "thin-section"


def display_image_name(filename: str) -> str:
    return Path(filename).stem.replace("_", " ")


def boxplot_values(frame: pd.DataFrame) -> list[np.ndarray]:
    values = [
        100.0 * frame.loc[frame["experience"] == level, "porosity"].to_numpy()
        for level in EXPERIENCE_LEVELS
    ]
    empty_levels = [
        level for level, level_values in zip(EXPERIENCE_LEVELS, values) if not len(level_values)
    ]
    if empty_levels:
        raise ValueError(f"No estimates found for experience levels: {empty_levels}")
    return values


def plot_boxplots(frame: pd.DataFrame, title: str, output_path: Path) -> Path:
    values = boxplot_values(frame)

    with publication_style():
        figure, axis = plt.subplots(figsize=(8, 6.4), constrained_layout=True)
        axis.boxplot(
            values,
            positions=EXPERIENCE_LEVELS,
            tick_labels=[str(level) for level in EXPERIENCE_LEVELS],
            widths=0.56,
            whis=1.5,
            showfliers=True,
            patch_artist=True,
            boxprops={"facecolor": "#aec7e8", "edgecolor": "#244a64", "linewidth": 1.4},
            medianprops={"color": "#8b1a1a", "linewidth": 2.1},
            whiskerprops={"color": "#244a64", "linewidth": 1.3},
            capprops={"color": "#244a64", "linewidth": 1.3},
            flierprops={
                "marker": "o",
                "markerfacecolor": "#244a64",
                "markeredgecolor": "#244a64",
                "markersize": 3.2,
                "alpha": 0.45,
            },
        )
        axis.set_title(textwrap.fill(title, width=48), pad=14)
        axis.set_xlabel("Experience level")
        axis.set_ylabel("Estimated porosity (%)")
        axis.set_xlim(0.5, 5.5)
        axis.set_ylim(0.0, 100.0)
        axis.set_yticks(np.arange(0.0, 101.0, 20.0))
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=100.0, decimals=0))
        axis.grid(axis="y", color="#d0d0d0", linewidth=0.8, alpha=0.75)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

        saved_path = save_pdf(figure, output_path)
        plt.close(figure)

    return saved_path


def generate_image_set_plots(image_set: ImageSet, output_dir: Path) -> list[Path]:
    frame = select_highest_resolution(load_estimates(image_set.results_path))
    output_paths: list[Path] = []

    for image_name in frame["image"].drop_duplicates():
        image_frame = frame[frame["image"] == image_name]
        output_path = output_dir / (
            f"{image_set.key}--{filename_slug(Path(image_name).stem)}"
            "--porosity-by-experience.pdf"
        )
        output_paths.append(
            plot_boxplots(
                image_frame,
                f"Individual porosity estimates - {display_image_name(image_name)}",
                output_path,
            )
        )

    grouped_path = output_dir / (
        f"{image_set.key}--all-thin-sections--porosity-by-experience.pdf"
    )
    output_paths.append(
        plot_boxplots(
            frame,
            f"Individual porosity estimates - {image_set.title}",
            grouped_path,
        )
    )
    return output_paths


def main() -> int:
    args = parse_args()
    image_sets = (
        ImageSet("article", "Original article thin sections", args.article_results),
        ImageSet("revision", "Final revision thin sections", args.revision_results),
    )

    output_paths: list[Path] = []
    for image_set in image_sets:
        output_paths.extend(generate_image_set_plots(image_set, args.output_dir))

    print(f"Wrote {len(output_paths)} PDF plots to {args.output_dir.resolve()}")
    for output_path in output_paths:
        print(f"  {output_path.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\n[ERROR] {exc}")
        raise SystemExit(1)
