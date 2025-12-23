#!/bin/bash
set -e  # stop on errors

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Cleaning temporary files..."

# Patterns to remove
patterns=(
  "__pycache__"
  "*.egg-info"
  "*win_*"
)

for pattern in "${patterns[@]}"; do
  echo "Removing $pattern ..."
  find . -name "$pattern" -exec rm -rf {} +
done

if [ ! -L ".venv" ]; then
    sudo rm -rf .venv
fi

echo "Cleanup done!"