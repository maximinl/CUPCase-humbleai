# HuggingFace Local Model Experiments

Systematic evaluation of Qwen3.5 models for the CUPCase diagnostic pipeline.
Experiments run on MIT HPC `mit_normal_gpu` partition:
- **v01-v04 (9B):** H100 GPUs (80GB VRAM) — 1 GPU per job
- **v05-v10 (27B):** H200 GPUs (141GB VRAM) — 1 GPU per job

Previous runs on L40S (46GB) OOM'd for 27B models. Switched to larger GPUs on 2026-03-18.

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

## Experiment Matrix (13 variants)

All on `mit_normal_gpu` partition, H200 GPUs (141GB VRAM). N=10, seed=42.

### Phase 1: Initial generation (v01-v05, completed)

Ran with same model as judge — results showed judge unreliability (v02 = 100% with thinking).

| ID | Main | Quant | Judge | Gen Thinking | Judge Thinking | GPU | What it tests |
|----|------|-------|-------|:---:|:---:|:---:|---------------|
| v01 | 9B | bf16 | 9B bf16 | off | off | H100 | 9B baseline |
| v02 | 9B | bf16 | 9B bf16 | **on** | **on** | H100 | Thinking (both gen+judge — **flawed: judge too lenient**) |
| v03 | 9B | 4bit | 9B 4bit | off | off | H100 | 4-bit quantization |
| v04 | 9B | 8bit | 9B 8bit | off | off | H100 | 8-bit quantization |
| v05 | 27B | 4bit | 9B bf16 | off | off | H200 | Larger model |

### Phase 2: Redesigned experiments (v06-v10)

**Key fixes:**
- Separated judge model from generation model (`--model-judge`, `--judge-thinking` flags)
- Judge always uses 27B bf16 with thinking OFF for reliable evaluation
- v06 re-judges v01-v05 CSVs with the better judge (no re-generation needed)
- v09 tests true ensemble (different main vs small models)

| ID | Main | Quant | Small | Judge | Gen Thinking | Judge Thinking | GPU | What it tests |
|----|------|-------|-------|-------|:---:|:---:|:---:|---------------|
| v06 | — | — | — | 27B bf16 | — | off | H200 | **REJUDGE** v01-v05 with reliable judge |
| v07 | 27B | bf16 | 27B bf16 | 27B bf16 | off | off | H200 | Best quality baseline (full precision) |
| v08 | 27B | bf16 | 27B bf16 | 27B bf16 | **on** | off | H200 | Does CoT help generation? (reliable judge) |
| v09 | 27B | bf16 | **9B bf16** | 27B bf16 | off | off | H200 | **True ensemble** (different models!) |
| v10 | 9B | bf16 | 9B bf16 | 27B bf16 | **on** | off | H200 | Cheapest good config? (9B+thinking, 27B judge) |

### Key comparisons (Phase 2)

- **Judge calibration:** v06 re-judges v01-v05 → reveals true generation quality
- **Thinking effect (reliable judge):** v07 vs v08 (27B), v01-rejudged vs v10 (9B)
- **Model size:** v10 (9B+think) vs v07 (27B no think) — is 9B+CoT competitive?
- **True ensemble:** v09 (27B+9B) vs v07 (27B alone) — does diversity help?
- **Full precision vs quantized:** v07 (27B bf16) vs v05-rejudged (27B 4bit)

### Phase 3: Pipeline fix experiments (v11-v13)

Each tests a specific fix from the GitHub issues on `maximinl/CUPCase-humbleai`.

| ID | Fix | Main | Small | Judge | Thinking | Diverse | Audit | max_tokens | Issue |
|----|-----|------|-------|-------|:---:|:---:|:---:|:---:|:---:|
| v11 | Thinking extraction | 27B bf16 | 27B bf16 | 27B bf16 | **on** | legacy | legacy | 4096 | [#1](https://github.com/maximinl/CUPCase-humbleai/issues/1) |
| v12 | Diverse candidates | 27B bf16 | **9B bf16** | 27B bf16 | off | **differential** | legacy | 1024 | [#2](https://github.com/maximinl/CUPCase-humbleai/issues/2) |
| v13 | Differential audit | 27B bf16 | **9B bf16** | 27B bf16 | off | **differential** | **differential** | 2048 | [#3](https://github.com/maximinl/CUPCase-humbleai/issues/3) |

### Key comparisons (Phase 3)

- **v11 vs v08**: Isolates thinking extraction fix (same config, fixed code)
- **v12 vs v09**: Isolates diverse candidate generation (same models, new candidate strategy)
- **v13 vs v12**: Isolates differential audit (same candidates, new audit prompt)
- **v13 vs v07**: Full pipeline improvement (all fixes vs best baseline)

### Memory budget (H200, 141GB VRAM)

- v06: 27B bf16 judge only ≈ 54GB
- v07: 27B bf16 × 2 (gen + judge reused) ≈ 54GB
- v08: 27B bf16 × 2 ≈ 54GB (thinking just uses more tokens, not more VRAM)
- v09: 27B bf16 + 9B bf16 + 27B judge (reuses main) ≈ 72GB
- v10: 9B bf16 × 2 + 27B bf16 judge ≈ 72GB
- v11: same as v08 ≈ 54GB (more tokens but same VRAM)
- v12: same as v09 ≈ 72GB (more model calls but same memory)
- v13: same as v09 ≈ 72GB

## Results (last updated: 2026-03-18)

### Phase 1 results (9B judge — unreliable, awaiting v06 rejudge)

| ID | Status | Easy Baseline | Easy Ensemble | Easy Audit | Easy Hybrid | Hard Baseline | Hard Ensemble | Hard Audit | Hard Hybrid | Notes |
|----|:------:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|-------|
| v01 | [X] | 50.0% | 50.0% | 50.0% | 50.0% | 60.0% | 60.0% | 60.0% | 60.0% | H100, 7s/case |
| v02 | [X] | ~~100%~~ | ~~100%~~ | ~~100%~~ | ~~100%~~ | ~~100%~~ | ~~100%~~ | ~~100%~~ | ~~100%~~ | **INVALID** — judge used thinking, too lenient |
| v03 | [X] | 50.0% | 50.0% | 50.0% | 50.0% | 50.0% | 50.0% | 50.0% | 50.0% | 4bit, 10s/case |
| v04 | [X] | 60.0% | 60.0% | 60.0% | 60.0% | 60.0% | 60.0% | 60.0% | 60.0% | 8bit, 25s/case |
| v05 | [X] | 70.0% | 70.0% | 70.0% | 70.0% | 50.0% | 50.0% | 50.0% | 50.0% | H200, 27B-4bit main, 25s/case |

### Phase 2 results (27B bf16 judge — reliable)

| ID | Status | Easy Baseline | Easy Ensemble | Easy Audit | Easy Hybrid | Hard Baseline | Hard Ensemble | Hard Audit | Hard Hybrid | Notes |
|----|:------:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|-------|
| v06 | [ ] | — | — | — | — | — | — | — | — | Rejudge of v01-v05 |
| v07 | [ ] | — | — | — | — | — | — | — | — | |
| v08 | [ ] | — | — | — | — | — | — | — | — | |
| v09 | [ ] | — | — | — | — | — | — | — | — | |
| v10 | [ ] | — | — | — | — | — | — | — | — | |

### Phase 3 results (pipeline fixes)

| ID | Status | Easy Baseline | Easy Ensemble | Easy Audit | Easy Hybrid | Hard Baseline | Hard Ensemble | Hard Audit | Hard Hybrid | Notes |
|----|:------:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|-------|
| v11 | [ ] | — | — | — | — | — | — | — | — | Issue #1: thinking fix |
| v12 | [ ] | — | — | — | — | — | — | — | — | Issue #2: diverse candidates |
| v13 | [ ] | — | — | — | — | — | — | — | — | Issue #3: differential audit |

### Status key

- `[ ]` = submitted / pending
- `[X]` = done, results recorded
- `[!]` = failed — see Notes column
- `[R]` = running

### Observations from Phase 1

1. **v02 results are INVALID.** When `--enable-thinking` was global, the judge also used thinking
   and became too lenient (100% across the board). Fixed in Phase 2 by separating
   `--enable-thinking` (generation) from `--judge-thinking` (judge).

2. **All 4 methods give identical scores** because main == small model in v01-v05.
   No ensemble diversity → no ensemble benefit. v09 fixes this with 27B main + 9B small.

3. **9B judge (no thinking) may be too strict.** 50-60% scores are lower than expected.
   v06 will re-judge with 27B to see if the 9B judge was the bottleneck.

4. **27B-4bit > 9B-bf16 on Easy** (70% vs 50%), but not on Hard (50% vs 60%).
   v08 tests if 27B + thinking can improve Hard performance.

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
