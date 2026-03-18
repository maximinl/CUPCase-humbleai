# HuggingFace Local Model Experiments

Systematic evaluation of Qwen3.5 models for the CUPCase diagnostic pipeline.
All experiments run on MIT HPC (L40S GPUs, 46GB VRAM each), `mit_preemptable` partition.

## Bug Fixes Applied (2026-03-18)

1. **Thinking token leak** — Qwen3.5 generates `<think>...</think>` blocks; previous code included
   thinking content in the diagnosis output. Fixed in `hf_client.py`: now strips thinking blocks
   and returns only the answer text after `</think>`.
2. **`raw` variable bug** — `hf_experiment.py:perform_audit()` used `raw` in the `except` block
   without initializing it, crashing when `client.completion()` itself raised an exception.
   Fixed by initializing `raw = None` before the try block.
3. **`enable_thinking` flag** — Added CLI `--enable-thinking` to control whether Qwen3.5 uses
   its chain-of-thought reasoning mode. When enabled, `max_tokens` should be increased to 2048+
   to leave room for the answer after thinking.

## Experiment Matrix (10 variants)

All use N=10 samples, seed=42, `mit_preemptable` partition.

| ID | Main Model | Main Quant | Judge Model | Judge Quant | GPUs | RAM | Thinking | max_tokens | What it tests |
|----|-----------|-----------|-------------|------------|:----:|:---:|:--------:|:----------:|---------------|
| v01 | Qwen3.5-9B | bf16 | Qwen3.5-9B | bf16 | 2 | 128G | off | 1024 | Clean baseline after thinking fix |
| v02 | Qwen3.5-9B | bf16 | Qwen3.5-9B | bf16 | 2 | 128G | **on** | 2048 | Does thinking improve 9B accuracy? |
| v03 | Qwen3.5-9B | 4bit | Qwen3.5-9B | 4bit | 1 | 96G | off | 1024 | 4-bit quantization impact on 9B |
| v04 | Qwen3.5-9B | 8bit | Qwen3.5-9B | 8bit | 1 | 96G | off | 1024 | 8-bit quantization impact on 9B |
| v05 | Qwen3.5-27B | 4bit | Qwen3.5-9B | bf16 | 2 | 196G | off | 1024 | Larger model, quantized, 9B judge |
| v06 | Qwen3.5-27B | 4bit | Qwen3.5-9B | bf16 | 2 | 196G | **on** | 2048 | Does thinking improve 27B accuracy? |
| v07 | Qwen3.5-27B | 8bit | Qwen3.5-9B | bf16 | 2 | 196G | off | 1024 | 8-bit 27B vs 4-bit 27B |
| v08 | Qwen3.5-27B | bf16 | Qwen3.5-9B | bf16 | 2 | 196G | off | 1024 | Full-precision 27B (best quality?) |
| v09 | Qwen3.5-27B | 4bit | Qwen3.5-27B | 4bit | 2 | 196G | off | 1024 | Does 27B judge outperform 9B judge? |
| v10 | Qwen3.5-27B | 8bit | Qwen3.5-27B | 8bit | 2 | 196G | off | 1024 | 8-bit 27B for both roles |

### Key comparisons

- **Quantization effect (9B):** v01 (bf16) vs v03 (4bit) vs v04 (8bit)
- **Quantization effect (27B):** v05 (4bit) vs v07 (8bit) vs v08 (bf16)
- **Thinking mode:** v01 vs v02 (9B), v05 vs v06 (27B)
- **Model size:** v01 (9B) vs v05 (27B-4bit) vs v08 (27B-bf16)
- **Judge quality:** v05 (9B judge) vs v09 (27B judge)

### Memory budget rationale

Previous runs OOM'd on L40S (46GB). Memory allocations use ~50% headroom:
- 9B bf16 ≈ 18GB × 2 models = 36GB → 2 GPUs (92GB available)
- 9B 4bit ≈ 5GB × 2 = 10GB → 1 GPU (46GB available)
- 27B 4bit + 9B bf16 ≈ 32GB → 2 GPUs (92GB available)
- 27B bf16 + 9B bf16 ≈ 72GB → 2 GPUs (92GB available)
- 27B 8bit × 2 ≈ 54GB → 2 GPUs (92GB available)

## Results (last updated: 2026-03-18)

| ID | Status | Easy Baseline | Easy Ensemble | Easy Audit | Easy Hybrid | Hard Baseline | Hard Ensemble | Hard Audit | Hard Hybrid | Notes |
|----|:------:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|-------|
| v01 | [ ] | — | — | — | — | — | — | — | — | |
| v02 | [ ] | — | — | — | — | — | — | — | — | |
| v03 | [ ] | — | — | — | — | — | — | — | — | |
| v04 | [ ] | — | — | — | — | — | — | — | — | |
| v05 | [ ] | — | — | — | — | — | — | — | — | |
| v06 | [ ] | — | — | — | — | — | — | — | — | |
| v07 | [ ] | — | — | — | — | — | — | — | — | |
| v08 | [ ] | — | — | — | — | — | — | — | — | |
| v09 | [ ] | — | — | — | — | — | — | — | — | |
| v10 | [ ] | — | — | — | — | — | — | — | — | |

### Status key

- `[ ]` = submitted / pending
- `[X]` = done, results recorded
- `[!]` = failed — see Notes column
- `[R]` = running

## Launch

```bash
cd /orcd/pool/005/sebasmos/code/CUPCase-humbleai

# Launch all 10
bash slurm-hf/launch_all.sh

# Or launch individually
sbatch slurm-hf/run_v01.sh

# Monitor
squeue -u $USER | grep hf-v
```

## Output

Each variant writes to `output-hf/vXX-{description}/`:
- `qwen35_easy_10.csv` — raw pipeline results on Easy dataset
- `qwen35_hard_10.csv` — raw pipeline results on Hard dataset
