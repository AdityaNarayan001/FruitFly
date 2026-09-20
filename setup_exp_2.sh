#!/bin/sh
# Run from any working directory; keep all dependencies inside this checkout.
set -eu
FFB_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if [ -n "${FFB_PYTHON:-}" ]; then
    exec "$FFB_PYTHON" "$FFB_ROOT/scripts/setup_exp_2.py" "$@"
fi
for FFB_CANDIDATE in python3.12 python3.13 python3.11 python3; do
    if command -v "$FFB_CANDIDATE" >/dev/null 2>&1 && "$FFB_CANDIDATE" -c 'import sys; sys.exit(not ((3,11) <= sys.version_info[:2] < (3,14)))' 2>/dev/null; then
        exec "$FFB_CANDIDATE" "$FFB_ROOT/scripts/setup_exp_2.py" "$@"
    fi
done
printf '%s\n' 'FruitFlyBrain needs Python 3.11, 3.12 or 3.13 with venv support.' 'Install Python from https://www.python.org/downloads/ and run this command again.' 'If it is outside PATH: FFB_PYTHON=/path/to/python3.12 sh setup.sh'
exit 1
