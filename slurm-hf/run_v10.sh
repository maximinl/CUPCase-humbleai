#!/bin/bash
# v10: 9B bf16 + thinking ON (generation) + 27B bf16 judge (thinking OFF)
# Purpose: Cheapest good config? 9B is fast — if thinking makes it competitive
#          with 27B, this is the best cost/accuracy tradeoff
#SBATCH --job-name=hf-v10
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-hf/logs/v10-%j.out
#SBATCH --error=slurm-hf/logs/v10-%j.err

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
    --model-main Qwen/Qwen3.5-9B \
    --model-small Qwen/Qwen3.5-9B \
    --model-judge Qwen/Qwen3.5-27B \
    --enable-thinking \
    --samples 10 --seed 42 \
    --max-tokens 2048 \
    --output-dir output-hf/v10-9bthink-27bjudge
