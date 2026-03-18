#!/bin/bash
# v13: 27B main + 9B small + 27B judge — DIFFERENTIAL AUDIT + DIVERSE CANDIDATES
# Purpose: Validate Issue #3 — audit generates counter-hypotheses, not limited to input
# Ref: https://github.com/maximinl/CUPCase-humbleai/issues/3
#SBATCH --job-name=hf-v13
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-hf/logs/v13-%j.out
#SBATCH --error=slurm-hf/logs/v13-%j.err

set -e
cd /orcd/pool/005/sebasmos/code/CUPCase-humbleai
mkdir -p slurm-hf/logs
module load cuda/12.4.0

for i in 1 2 3; do
    python3.11 -m pip install --user \
        torch transformers accelerate huggingface_hub \
        bitsandbytes pandas tqdm nest_asyncio python-dotenv requests \
        2>&1 | tail -3 && break
    echo "Install attempt $i failed, retrying..." && sleep 5
done

python3.11 -c "import torch; print('torch OK:', torch.__version__)" || { echo "FATAL: torch"; exit 1; }
set -a; [ -f .env ] && source .env; set +a
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

python3.11 hf_experiment.py \
    --model-main Qwen/Qwen3.5-27B \
    --model-small Qwen/Qwen3.5-9B \
    --model-judge Qwen/Qwen3.5-27B \
    --diverse-mode differential \
    --audit-mode differential \
    --samples 10 --seed 42 \
    --max-tokens 2048 \
    --output-dir output-hf/v13-27b9b-diffaudit-27bjudge
