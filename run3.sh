#!/usr/bin/env bash
# One command: ./run3.sh  -> http://localhost:8501
set -euo pipefail
cd "$(dirname "$0")"
[ -x .venv/bin/python ] || python3 -m venv .venv
.venv/bin/pip3 install -q -r app/requirements.txt
export PYDANTIC_AI_NO_BANNER=1
exec .venv/bin/streamlit run app/app.py --server.headless true --browser.gatherUsageStats false --server.port "${PORT:-8501}"
