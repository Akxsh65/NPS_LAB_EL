#!/usr/bin/env bash
# Thin wrapper — use run_architecture_from_stage2.sh (loads winner from stage2_winner.env).
set -euo pipefail
cd "$(dirname "$0")/.."
exec bash scripts/run_architecture_from_stage2.sh
