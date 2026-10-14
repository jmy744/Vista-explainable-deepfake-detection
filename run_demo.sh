#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  python3.12 -m venv .venv
fi
if [ ! -f .venv/demo-ready.txt ]; then
  .venv/bin/python -m pip install -r requirements.txt
  touch .venv/demo-ready.txt
fi
.venv/bin/python -m streamlit run app.py
