#!/bin/bash
# Run ruff lint checks (configured in pyproject.toml).
# Usage: ./scripts/lint.sh [--fix]
#   --fix  apply ruff's safe automatic fixes

set -euo pipefail
cd "$(dirname "$0")/.."

if [ "${1:-}" = "--fix" ]; then
    echo "Linting with ruff (applying fixes)..."
    uv run ruff check --fix .
else
    echo "Linting with ruff..."
    uv run ruff check .
fi
