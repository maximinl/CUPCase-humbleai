#!/bin/bash
# v06: 27B 4bit main + 9B bf16 judge — 2 GPU — thinking mode ON
#SBATCH --job-name=hf-v06
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=196G
#SBATCH --time=08:00:00
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

python3.11 hf_experiment.py \
    --model-main Qwen/Qwen3.5-27B \
    --model-small Qwen/Qwen3.5-9B \
    --quantize-main 4bit \
    --samples 10 --seed 42 \
    --max-tokens 2048 \
    --enable-thinking \
    --output-dir output-hf/v06-27b4bit-9b-think
