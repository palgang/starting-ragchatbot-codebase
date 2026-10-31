#!/bin/bash
# Auto-format all Python code with black (rewrites files in place).
# Usage: ./scripts/format.sh [--check]
#   --check  report files that would change without modifying them

set -euo pipefail
cd "$(dirname "$0")/.."

if [ "${1:-}" = "--check" ]; then
    echo "Checking formatting with black..."
    uv run black --check .
else
    echo "Formatting Python code with black..."
    uv run black .
fi
