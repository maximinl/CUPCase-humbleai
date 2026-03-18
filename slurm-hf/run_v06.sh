#!/bin/bash
# v06: REJUDGE v01-v05 with 27B bf16 judge (thinking OFF)
# Purpose: Calibrate judge — find which generation config is actually best
# All generation already done; this only runs the judge on existing CSVs
#SBATCH --job-name=hf-v06
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-hf/logs/v06-%j.out
#SBATCH --error=slurm-hf/logs/v06-%j.err

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

python3.11 rejudge.py \
    --judge-model Qwen/Qwen3.5-27B \
    --csv-dir \
        output-hf/v01-9b-bf16 \
        output-hf/v02-9b-bf16-think \
        output-hf/v03-9b-4bit \
        output-hf/v04-9b-8bit \
        output-hf/v05-27b4bit-9b
