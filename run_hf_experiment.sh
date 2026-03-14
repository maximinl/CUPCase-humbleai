#!/bin/bash
#SBATCH --job-name=hf-qwen35-full
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-hf-full-%j.out
#SBATCH --error=slurm-hf-full-%j.err

# ============================================================
# Full experiment: 3 Qwen3.5 variants x 2 datasets x 10 samples
# ============================================================

set -e
cd /orcd/pool/005/sebasmos/code/CUPCase-humbleai

module load cuda/12.4.0

python3.11 -m pip install --user transformers torch accelerate huggingface_hub \
    bitsandbytes pandas tqdm nest_asyncio python-dotenv requests 2>&1 | tail -5

set -a; source .env; set +a

echo "============================================"
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "============================================"

SAMPLES=10
SEED=42

# ---- Qwen3.5-9B (dense, small) ----
echo -e "\n>>> Qwen3.5-9B (bf16)\n"
python3.11 hf_experiment.py \
    --model Qwen/Qwen3.5-9B \
    --samples $SAMPLES --seed $SEED --output-dir output-hf

# ---- Qwen3.5-35B-A3B (MoE, 3B active) ----
echo -e "\n>>> Qwen3.5-35B-A3B (bf16, MoE)\n"
python3.11 hf_experiment.py \
    --model Qwen/Qwen3.5-35B-A3B \
    --samples $SAMPLES --seed $SEED --output-dir output-hf

# ---- Qwen3.5-27B (dense, strongest) ----
echo -e "\n>>> Qwen3.5-27B (bf16, dense)\n"
python3.11 hf_experiment.py \
    --model Qwen/Qwen3.5-27B \
    --samples $SAMPLES --seed $SEED --output-dir output-hf

echo -e "\n============================================"
echo "All experiments complete! Results in output-hf/"
echo "============================================"
