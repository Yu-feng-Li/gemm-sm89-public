#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
"${PYTHON:-python3}" -m venv .venv
.venv/bin/python -m pip install --no-cache-dir -r requirements.txt
