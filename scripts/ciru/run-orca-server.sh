#!/usr/bin/env bash
set -euo pipefail
export MODEL_VARIANT=Orca
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run-server.sh" "$@"
