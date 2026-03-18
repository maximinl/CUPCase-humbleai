#!/bin/bash
#SBATCH --job-name=hf-dual
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-hf-dual-%j.out
#SBATCH --error=slurm-hf-dual-%j.err

set -e
cd /orcd/pool/005/sebasmos/code/CUPCase-humbleai
module load cuda/12.4.0

for i in 1 2 3; do
    python3.11 -m pip install --user \
        torch transformers accelerate huggingface_hub \
        bitsandbytes pandas tqdm nest_asyncio python-dotenv requests \
        2>&1 | tail -3 && break
    echo "Install attempt $i failed, retrying..."
    sleep 5
done

python3.11 -c "import torch; print('torch OK:', torch.__version__)" || { echo "FATAL: torch not installed"; exit 1; }

set -a; source .env; set +a

echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

python3.11 hf_experiment.py \
    --model-main Qwen/Qwen3.5-27B \
    --model-small Qwen/Qwen3.5-9B \
    --quantize-main 4bit \
    --samples 10 --seed 42 \
    --output-dir output-hf-dual
