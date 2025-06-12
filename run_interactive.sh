#!/bin/bash
source .env
if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN environment variable is not set."
    echo "Please create a .env file in the current directory with the following content:"
    echo "export HF_TOKEN=\"your huggingface token\""
    exit 1
fi
export HF_HOME=./cache
export HUGGINGFACE_HUB_CACHE=hub
export TRANSFORMERS_CACHE=transformers
export HF_DATASETS_CACHE=datasets
source .venv/bin/activate
python sd3.5_diffusers_control.py