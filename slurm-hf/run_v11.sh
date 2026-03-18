#!/bin/bash
# v11: 27B bf16 + thinking (FIXED extraction) + 27B bf16 judge (thinking OFF)
# Purpose: Validate Issue #1 — thinking mode should produce real diagnoses (not garbage)
# Ref: https://github.com/maximinl/CUPCase-humbleai/issues/1
#SBATCH --job-name=hf-v11
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-hf/logs/v11-%j.out
#SBATCH --error=slurm-hf/logs/v11-%j.err

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
    --model-small Qwen/Qwen3.5-27B \
    --model-judge Qwen/Qwen3.5-27B \
    --enable-thinking \
    --samples 10 --seed 42 \
    --max-tokens 4096 \
    --output-dir output-hf/v11-27bbf16-thinkfix-27bjudge
