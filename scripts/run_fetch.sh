#!/usr/bin/env bash
# Run the data fetcher directly on docker-host (no Docker needed if vnstock installed).
# Useful for quick re-runs or testing specific symbols.
#
# Usage:
#   ./scripts/run_fetch.sh                          # all HOSE + HNX
#   ./scripts/run_fetch.sh --symbols VIC,HPG,SHB   # specific symbols
#   ./scripts/run_fetch.sh --no-resume              # full re-fetch

set -euo pipefail

export DATA_DIR="${DATA_DIR:-/root/reports/vn_training_data}"
export PYTHONUNBUFFERED=1

python -m vn_chronos.fetch_data --exchange HOSE HNX "$@"
