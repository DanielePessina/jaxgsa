#!/bin/bash
set -euo pipefail
# Autoresearch benchmark for jaxgsa GSA kernels.
# Runs the full workload matrix (one fresh subprocess per case), writes the
# per-case table to .auto/bench_latest.md (view in tmux), and emits METRIC
# lines for the loop.
cd "$(dirname "$0")/.."
uv run .auto/bench_gsa.py --repeats 3
