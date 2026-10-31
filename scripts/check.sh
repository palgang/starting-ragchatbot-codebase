#!/bin/bash
# Run all quality checks without modifying any files. Intended for CI and
# pre-commit use. Exits non-zero if any check fails.

set -uo pipefail
cd "$(dirname "$0")/.."

failed=0

echo "== black (format check) =="
uv run black --check . || failed=1
echo

echo "== ruff (lint) =="
uv run ruff check . || failed=1
echo

if [ "$failed" -ne 0 ]; then
    echo "Quality checks FAILED. Run ./scripts/format.sh and ./scripts/lint.sh --fix,"
    echo "then fix any remaining lint errors manually."
    exit 1
fi

echo "All quality checks passed."
