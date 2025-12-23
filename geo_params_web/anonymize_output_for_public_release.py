"""
WARNING:
This script irreversibly anonymizes user interaction data by removing
personal identifiers and free-text feedback.

Before modifying any files, a full backup of the output directory is
created in a sibling folder named 'output.uncommitted'.

Run this script ONLY on data intended for public release.
Always keep a private backup of raw data before running.
"""

import json
import shutil
from pathlib import Path
import sys

print("""
WARNING:
This script irreversibly anonymizes user interaction data by removing
personal identifiers and free-text feedback.

A full backup of the output directory will be created before any changes
are applied.

Run this script ONLY on data intended for public release.
Always keep a private backup of raw data before running.
""")
input("Press ENTER to continue, or CTRL+C to abort.")

OUTPUT_DIR = Path("static/output")
BACKUP_DIR = OUTPUT_DIR.with_name(f"{OUTPUT_DIR.name}.uncommitted")

# --- Sanity checks ---
if not OUTPUT_DIR.exists():
    raise RuntimeError(f"Output directory not found: {OUTPUT_DIR}")

if BACKUP_DIR.exists():
    print(f"ERROR: Backup directory already exists: {BACKUP_DIR}")
    print(
        "To avoid overwriting raw data, the procedure was aborted.\n"
        "Please inspect or remove the existing backup directory manually "
        "before running this script again."
    )
    sys.exit(1)

# --- Create backup ---
print(f"Creating backup: {BACKUP_DIR.resolve()}")
shutil.copytree(OUTPUT_DIR, BACKUP_DIR)
print("Backup completed successfully.")
print("-" * 60)

removed_reviews = 0
anonymized_options = 0
removed_test_sessions = 0


def is_test_user(options):
    """
    Detect developer test sessions.
    This is intentionally conservative.
    """
    name = options.get("user", {}).get("name")
    return isinstance(name, str) and "test" in name.lower()


# --- Process all options.json files, anywhere ---
for options_path in OUTPUT_DIR.rglob("options.json"):
    session_dir = options_path.parent

    with open(options_path, "r", encoding="utf-8") as f:
        options = json.load(f)

    # Remove entire test sessions (recommended)
    if is_test_user(options):
        shutil.rmtree(session_dir)
        removed_test_sessions += 1
        print(f"Removed test session: {session_dir.relative_to(OUTPUT_DIR)}")
        continue

    user = options.setdefault("user", {})

    # Remove personal identifiers
    user["email"] = None
    user["name"] = None

    # Force anonymization flag
    user["anonymize"] = True
    options["user"] = user

    with open(options_path, "w", encoding="utf-8") as f:
        json.dump(options, f, indent=2, ensure_ascii=False)

    anonymized_options += 1


# --- Remove all free-text review files, anywhere ---
for review_path in OUTPUT_DIR.rglob("end_review.txt"):
    review_path.unlink()
    removed_reviews += 1


print("-" * 60)
print("Cleanup summary:")
print(f"  Backup directory created     : {BACKUP_DIR}")
print(f"  Anonymized options.json files: {anonymized_options}")
print(f"  Removed end_review.txt files : {removed_reviews}")
print(f"  Removed test sessions        : {removed_test_sessions}")
print("Done.")