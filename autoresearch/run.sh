#!/bin/bash
# ScisTreeCNA AutoResearch Launch Script
# =======================================
# Usage: bash autoresearch/run.sh [tag]
#
# This script:
# 1. Creates a fresh branch autoresearch/<tag>
# 2. Runs the baseline evaluation
# 3. Launches Claude Code in headless mode to iterate autonomously
#
# Based on Karpathy's autoresearch pattern.

set -e

TAG="${1:-$(date +%b%d | tr '[:upper:]' '[:lower:]')}"
BRANCH="autoresearch/$TAG"

echo "=========================================="
echo "ScisTreeCNA AutoResearch"
echo "=========================================="
echo "Tag: $TAG"
echo "Branch: $BRANCH"
echo ""

# Step 1: Create branch
echo "[1/3] Creating branch $BRANCH..."
git checkout -b "$BRANCH" 2>/dev/null || git checkout "$BRANCH"

# Step 2: Run baseline
echo "[2/3] Running baseline evaluation..."
python autoresearch/prepare.py 2>&1 | tee autoresearch/baseline.log
BASELINE=$(grep "^RESULT:" autoresearch/baseline.log | head -1)
echo ""
echo "Baseline: $BASELINE"
echo ""

# Step 3: Record baseline in results.tsv
COMMIT=$(git rev-parse --short HEAD)
TIME=$(echo "$BASELINE" | grep -oP 'total_time_s=\K[0-9.]+')
echo -e "0\t$COMMIT\tbaseline\t$TIME\t-15164.22\t0.0000\t0.9939\tPASS" >> autoresearch/results.tsv

# Step 4: Launch Claude Code
echo "[3/3] Launching Claude Code autoresearch loop..."
echo "Press Ctrl+C to stop the loop at any time."
echo ""

claude -p "Read autoresearch/program.md and begin the autoresearch experiment loop. Start by reading the current code state, reviewing OPTIMIZATION_REPORT.md for prior work, then run experiments. NEVER STOP until interrupted. The baseline is: $BASELINE"
