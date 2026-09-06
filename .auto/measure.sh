#!/bin/bash
set -euo pipefail
# Autoresearch benchmark for jaxgsa GSA kernels.
# Runs the workload matrix (one fresh subprocess per case), writes the
# per-case table to .auto/bench_latest.md (view in tmux), and emits METRIC
# lines for the loop. An optional first argument is a comma-separated case
# filter (e.g. ./.auto/measure.sh ot_small,ot_slices) for method-scoped
# sessions; without it the full matrix runs.
cd "$(dirname "$0")/.."
if [ $# -ge 1 ]; then
    uv run .auto/bench_gsa.py --repeats 9 --cases "$1"
else
    uv run .auto/bench_gsa.py --repeats 9
fi
