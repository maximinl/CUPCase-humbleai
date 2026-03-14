#!/bin/bash
#SBATCH --job-name=hf-27b
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-hf-27b-%j.out
#SBATCH --error=slurm-hf-27b-%j.err

# MEDIUM: Qwen3.5-27B for everything (~56GB VRAM)
# Single model, stronger reasoning

set -e
cd /orcd/pool/005/sebasmos/code/CUPCase-humbleai
module load cuda/12.4.0

python3.11 -m pip install --user --upgrade \
    transformers torch accelerate huggingface_hub \
    bitsandbytes pandas tqdm nest_asyncio python-dotenv requests \
    2>&1 | tail -5

set -a; source .env; set +a

echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

python3.11 hf_experiment.py \
    --model-main Qwen/Qwen3.5-27B \
    --model-small Qwen/Qwen3.5-27B \
    --samples 10 --seed 42 \
    --output-dir output-hf-27b
