#!/bin/sh
set -eu

repo_dir=$(CDPATH='' cd -P "$(dirname "$0")/.." && pwd)
exec "$repo_dir/run.sh" "$@"
