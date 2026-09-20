#!/bin/sh
# Reuse the pinned installer; launch the read-only explorer on its own port.
set -eu
FFB_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec sh "$FFB_ROOT/setup.sh" --app explorer "$@"
