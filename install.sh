#!/bin/bash
python3 -m venv .venv
source .venv/bin/activate
source .env
if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN environment variable is not set"
    exit 1
fi
export HF_HOME=./cache
export HUGGINGFACE_HUB_CACHE=hub
export TRANSFORMERS_CACHE=transformers
export HF_DATASETS_CACHE=datasets
pip install --cache-dir=.venv/pip-cache -r requirements.txt
echo "Running once to download all models"
python sd3.5_diffusers_control.py