#!/usr/bin/env bash
set -euo pipefail
cd /workspace/Data-Dash-Off-Market
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r part_b/requirements.txt -r part_b/report_requirements.txt -r part_b/test_requirements.txt
MPLCONFIGDIR=/tmp/part_b_matplotlib .venv/bin/python -m pytest part_b/tests -q
