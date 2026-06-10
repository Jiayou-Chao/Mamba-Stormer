#!/bin/bash
# Run the default Transformer vs BiMamba multi-resolution benchmark sweep.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

cd "$PROJECT_ROOT"

python scripts/comparison/transformer_vs_mamba/benchmark_resolution_sweep.py "$@"
