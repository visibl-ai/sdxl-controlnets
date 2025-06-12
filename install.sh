#!/bin/bash
python3 -m venv .venv
source .venv/bin/activate
pip install --cache-dir=.venv/pip-cache -r requirements.txt
echo "Running once to download all models"
./run_interactive.sh