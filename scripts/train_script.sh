#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Train + evaluate one evidence-span supervised LoRA model.
# All experiment settings are read by the Python scripts from config.yaml.
#
# Usage:
#   bash scripts/train_script.sh
#
# Optional:
#   CONFIG=config.myserver.yaml bash scripts/train_script.sh
# ============================================================

CONFIG="${CONFIG:-config.yaml}"
export PYTHONPATH="${PYTHONPATH:-.}"
export SPAN_CONFIG="${CONFIG}"

echo ">> [1/2] Running Training with ${CONFIG}"
python span_citation/training/05.lora_tuning.py --config "${CONFIG}"

echo ">> [2/2] Running Evaluation with ${CONFIG}"
python -u span_citation/training/llama_evidence.py --config "${CONFIG}"
