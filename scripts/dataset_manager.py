"""Download and verify the public GeoParams research datasets."""

from __future__ import annotations

import argparse
import hashlib
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import gdown


MANIFEST = Path(
    os.environ.get(
        "GEO_PARAMS_DATASET_MANIFEST",
        Path(__file__).with_name(".dataset-manifest.tsv"),
    )
)
DATA_ROOT = Path(os.environ.get("GEO_PARAMS_DATA_ROOT", "/datasets"))
APP_PREFIX = "article_thin_sections/"


@dataclass(frozen=True)
class DatasetFile:
    sha256: str
    size: int
    drive_id: str
    relative_path: PurePosixPath

    @property
    def target(self) -> Path:
        return DATA_ROOT.joinpath(*self.relative_path.parts)


def load_manifest(scope: str) -> list[DatasetFile]:
    entries: list[DatasetFile] = []
    for line_number, raw_line in enumerate(
        MANIFEST.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line or raw_line.startswith("#"):
            continue
        fields = raw_line.split("\t", 3)
        if len(fields) != 4:
            raise ValueError(f"Invalid manifest line {line_number}")
        expected_hash, size_text, drive_id, path_text = fields
        relative_path = PurePosixPath(path_text)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"Unsafe manifest path: {path_text}")
        if len(expected_hash) != 64:
            raise ValueError(f"Invalid SHA-256 on line {line_number}")
        if scope == "app" and not path_text.startswith(APP_PREFIX):
            continue
        entries.append(
            DatasetFile(
                sha256=expected_hash,
                size=int(size_text),
                drive_id=drive_id,
                relative_path=relative_path,
            )
        )
    return entries


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_valid(entry: DatasetFile, path: Path | None = None) -> bool:
    candidate = path or entry.target
    return (
        candidate.is_file()
        and candidate.stat().st_size == entry.size
        and sha256_file(candidate) == entry.sha256
    )


def verify(entries: list[DatasetFile]) -> int:
    valid = missing = invalid = 0
    for entry in entries:
        target = entry.target
        display = entry.relative_path.as_posix()
        if not target.exists():
            print(f"MISSING  {display}")
            missing += 1
        elif is_valid(entry):
            print(f"OK       {display}")
            valid += 1
        else:
            print(f"INVALID  {display}")
            invalid += 1
    print(f"\nVerified: {valid}; missing: {missing}; invalid: {invalid}")
    return 0 if missing == 0 and invalid == 0 else 2


def preserve_invalid(path: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = path.with_name(f"{path.name}.invalid-{timestamp}")
    suffix = 1
    while backup.exists():
        backup = path.with_name(f"{path.name}.invalid-{timestamp}-{suffix}")
        suffix += 1
    path.rename(backup)
    return backup


def download(entries: list[DatasetFile], replace_invalid: bool) -> int:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    failed = 0
    for index, entry in enumerate(entries, start=1):
        target = entry.target
        display = entry.relative_path.as_posix()
        target.parent.mkdir(parents=True, exist_ok=True)

        if is_valid(entry):
            print(f"[{index}/{len(entries)}] Already verified: {display}")
            continue

        if target.exists():
            if not replace_invalid:
                print(f"[{index}/{len(entries)}] Invalid; not replaced: {display}")
                failed += 1
                continue
            backup = preserve_invalid(target)
            print(f"Preserved invalid file as: {backup.name}")

        partial = target.with_name(f".{target.name}.partial")
        print(f"[{index}/{len(entries)}] Downloading: {display}")
        if partial.exists() and partial.stat().st_size >= entry.size:
            backup = preserve_invalid(partial)
            print(f"Preserved unusable partial file as: {backup.name}")
        try:
            result = gdown.download(
                id=entry.drive_id,
                output=os.fspath(partial),
                quiet=False,
                resume=True,
            )
        except Exception as error:  # gdown raises backend-specific errors
            print(f"Download failed: {display} ({error})")
            failed += 1
            continue
        if result is None or not partial.exists():
            print(f"Download failed: {display}")
            failed += 1
            continue
        if not is_valid(entry, partial):
            print(f"Hash verification failed; kept: {partial.name}")
            failed += 1
            continue
        os.replace(partial, target)
        print(f"Verified SHA-256: {display}")

    if failed:
        print(f"\nCompleted with {failed} unresolved file(s).")
        return 3
    print("\nAll selected dataset files are present and verified.")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("verify", "download"):
        child = subparsers.add_parser(command)
        child.add_argument("--scope", choices=("app", "all"), default="all")
        if command == "download":
            child.add_argument("--replace-invalid", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    entries = load_manifest(args.scope)
    if args.command == "verify":
        return verify(entries)
    return download(entries, args.replace_invalid)


if __name__ == "__main__":
    raise SystemExit(main())
