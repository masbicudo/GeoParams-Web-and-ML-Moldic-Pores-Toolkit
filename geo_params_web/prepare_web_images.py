"""
Prepare image files used by the web data-collection app.

The source images are read from the repository-level datasets directory.
Derived images are written to static/imgs_sections, which is treated as a
local cache for the Flask app and notebooks.
"""

from __future__ import annotations

import glob
import json
import os
import shutil
from pathlib import Path

from libs import images


PROJECT_DIR = Path(__file__).resolve().parent
REPO_DIR = PROJECT_DIR.parent
SOURCE_DIR = REPO_DIR / "datasets" / "article_thin_sections"
TARGET_DIR = PROJECT_DIR / "static" / "imgs_sections"
PERCENTAGES = ("12.5", "25", "50")


def load_metadata() -> dict:
    metadata_file = TARGET_DIR / "metadata.json"
    if not metadata_file.exists():
        raise FileNotFoundError(f"Image metadata was not found: {metadata_file}")
    with metadata_file.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def copy_original_images(metadata: dict) -> None:
    missing_files: list[str] = []
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    for filename in metadata:
        source = SOURCE_DIR / filename
        target = TARGET_DIR / filename

        if not source.exists():
            missing_files.append(filename)
            continue

        if target.exists():
            print(f"Already exists, skipping: {target}")
            continue

        print(f"Copying: {source} -> {target}")
        shutil.copy2(source, target)

    if missing_files:
        missing_list = "\n".join(f"  {name}" for name in missing_files)
        raise FileNotFoundError(
            "Required thin-section image files were not found.\n\n"
            f"Expected source directory:\n  {SOURCE_DIR}\n\n"
            "Download the public dataset from Google Drive and place it in "
            "the repository's datasets directory.\n\n"
            f"Missing files:\n{missing_list}"
        )


def resize_original_images(metadata: dict) -> None:
    for filename in metadata:
        for percentage in PERCENTAGES:
            target = TARGET_DIR / percentage / filename
            if target.exists():
                print(f"Already exists, skipping: {target}")
                continue

            print(f"Resizing {filename} to {percentage}%")
            images.resize_image_file(
                str(TARGET_DIR / filename),
                str(target),
                float(percentage),
            )


def is_cut_info_valid(cut_info: dict) -> bool:
    return all(key in cut_info for key in ["x", "y", "width", "height"])


def make_cropped(cut_info: dict, src_filename: str, dst_filename: str, path: str) -> bool:
    source_file = TARGET_DIR / src_filename
    target_file = Path(path) / dst_filename
    if not source_file.exists():
        print(f"    Source file not found, skipping: {source_file}")
        return False
    if target_file.exists():
        print(f"    Target file already exists, skipping: {target_file}")
        return False
    images.crop_image_file(
        str(source_file),
        str(target_file),
        cut_info["x"],
        cut_info["y"],
        cut_info["width"],
        cut_info["height"],
    )
    return True


def make_cuts() -> None:
    folder = TARGET_DIR / "cuts"
    for file in glob.glob(os.path.join(folder, "**", "cut-metadata.json"), recursive=True):
        print(f"Found: {file}")
        with open(file, "r", encoding="utf-8") as fp:
            data = json.load(fp)

        path = os.path.dirname(file)
        for key, obj in data.items():
            print(f"  Key: {key}")
            if is_cut_info_valid(obj):
                if make_cropped(obj, key, key, path):
                    print(f"    Successfully cropped: {key}")
                continue

            any_found = False
            for key2, obj2 in obj.items():
                dst_filename = key2
                if is_cut_info_valid(obj2):
                    if not dst_filename.endswith((".jpg", ".jpeg", ".png")):
                        dst_filename += os.path.splitext(key)[1]
                    if make_cropped(obj2, key, dst_filename, path):
                        print(f"    Successfully cropped: {dst_filename}")
                    any_found = True
                else:
                    print(f"    Missing cutting info: {key2}")
            if not any_found:
                print(f"    Missing cutting info: {key}")


def iter_cut_filenames(data: dict) -> list[str]:
    filenames: list[str] = []
    for source_filename, cut_info in data.items():
        if is_cut_info_valid(cut_info):
            filenames.append(source_filename)
            continue

        extension = os.path.splitext(source_filename)[1]
        for cut_name, nested_cut_info in cut_info.items():
            if not is_cut_info_valid(nested_cut_info):
                continue
            if cut_name.endswith((".jpg", ".jpeg", ".png")):
                filenames.append(cut_name)
            else:
                filenames.append(f"{cut_name}{extension}")
    return filenames


def resize_cuts() -> None:
    folder = TARGET_DIR / "cuts"
    for file in glob.glob(os.path.join(folder, "**", "cut-metadata.json"), recursive=True):
        print(f"Found: {file}")
        with open(file, "r", encoding="utf-8") as fp:
            data = json.load(fp)

        path = os.path.dirname(file)
        for filename in iter_cut_filenames(data):
            source_file = Path(path) / filename
            for percentage in PERCENTAGES:
                target_file = Path(path) / percentage / filename
                if not source_file.exists():
                    print(f"    Source file not found, skipping: {source_file}")
                    continue
                if target_file.exists():
                    print(f"    Target file already exists, skipping: {target_file}")
                    continue
                print(f"    Resizing to {percentage}: {filename}")
                images.resize_image_file(str(source_file), str(target_file), float(percentage))


def main() -> int:
    metadata = load_metadata()
    copy_original_images(metadata)
    resize_original_images(metadata)
    make_cuts()
    resize_cuts()
    print("Image preparation completed successfully.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\n[ERROR] {exc}")
        raise SystemExit(1)
