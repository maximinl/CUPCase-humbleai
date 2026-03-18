#!/bin/bash
# Launch all 10 HuggingFace experiment variants
# Usage: bash slurm-hf/launch_all.sh [--dry-run]

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

echo "============================================"
echo "HuggingFace Experiment Suite — 10 Variants"
echo "============================================"
echo ""

SUBMITTED=0
for script in "$SCRIPT_DIR"/run_v*.sh; do
    name=$(basename "$script" .sh)
    if [[ "$DRY_RUN" == true ]]; then
        echo "[DRY ] $name"
    else
        JOB_ID=$(sbatch "$script" | awk '{print $NF}')
        echo "[SUB ] $name -> JobID $JOB_ID"
        SUBMITTED=$((SUBMITTED + 1))
    fi
done

echo ""
echo "============================================"
if [[ "$DRY_RUN" == true ]]; then
    echo "Dry run — no jobs submitted"
else
    echo "Submitted $SUBMITTED jobs"
fi
echo "Monitor: squeue -u \$USER | grep hf-v"
echo "============================================"
