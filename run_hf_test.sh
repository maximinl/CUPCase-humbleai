#!/bin/bash
#SBATCH --job-name=hf-qwen35
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-hf-qwen35-%j.out
#SBATCH --error=slurm-hf-qwen35-%j.err

# ============================================================
# Qwen3.5 Full Pipeline Test
# Main model: Qwen3.5-27B (diagnosis, ~56GB)
# Small model: Qwen3.5-9B  (ensemble + judge, ~20GB)
# Total VRAM: ~76GB — fits on H200 (130GB)
# ============================================================

set -e
cd /orcd/pool/005/sebasmos/code/CUPCase-humbleai

module load cuda/12.4.0

# Install deps (cached after first run)
python3.11 -m pip install --user \
    transformers torch accelerate huggingface_hub \
    bitsandbytes pandas tqdm nest_asyncio python-dotenv requests \
    2>&1 | tail -5

# Load env (HF_TOKEN, etc.)
set -a; source .env; set +a

echo "============================================"
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "Python: $(python3.11 --version)"
echo "============================================"

# Run full pipeline: Baseline + Ensemble + Audit + Hybrid
# Main: Qwen3.5-27B | Ensemble+Judge: Qwen3.5-9B | 10 samples
python3.11 hf_experiment.py \
    --model-main Qwen/Qwen3.5-27B \
    --model-small Qwen/Qwen3.5-9B \
    --samples 10 \
    --seed 42 \
    --output-dir output-hf

echo ""
echo "============================================"
echo "COMPLETE — Results in output-hf/"
echo "============================================"
