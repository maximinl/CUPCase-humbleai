#!/bin/bash
# v08-n100: 27B bf16 generation WITH thinking + 27B bf16 judge — N=100
# Purpose: Does CoT help 27B at scale?
#SBATCH --job-name=hf-v08n
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=18:00:00
#SBATCH --output=slurm-hf/logs/v08-n100-%j.out
#SBATCH --error=slurm-hf/logs/v08-n100-%j.err

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
    --samples 100 --seed 42 \
    --max-tokens 2048 \
    --output-dir output-hf/v08-n100-27bbf16-think-27bjudge
