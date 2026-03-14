#!/bin/bash
#SBATCH --job-name=hf-dual
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=slurm-hf-dual-%j.out
#SBATCH --error=slurm-hf-dual-%j.err

# BEST: Qwen3.5-27B (main) + Qwen3.5-9B (ensemble+judge)
# ~76GB VRAM total — real ensemble with two different models

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
    --model-small Qwen/Qwen3.5-9B \
    --samples 10 --seed 42 \
    --output-dir output-hf-dual
