#!/usr/bin/env bash
#
# Run experiments for the LLM Council Router evaluation.
# Usage: bash experiments/run_experiments.sh [--live]
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

LIVE_FLAG="${1:---dummy}"

echo "========================================"
echo " LLM Council Router — Experiment Suite"
echo "========================================"

echo ""
echo "[1/4] Unit tests"
python -m pytest tests/test_pipeline.py -v --tb=short || python -m unittest tests/test_pipeline.py -v

echo ""
echo "[2/4] Pipeline evaluation (dummy mode)"
python experiments/evaluate.py --mode pipeline --num-trials 5 --save $LIVE_FLAG

echo ""
echo "[3/4] Baseline comparison"
python experiments/evaluate.py --mode baseline-vs-pipeline --num-trials 3 --save $LIVE_FLAG

echo ""
echo "[4/4] Interactive demo (dummy mode, 3 sample prompts)"
python src/main.py --demo $LIVE_FLAG

echo ""
echo "========================================"
echo " All experiments complete."
echo " Results saved to results/"
echo "========================================"
