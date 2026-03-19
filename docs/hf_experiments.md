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

## Results (last updated: 2026-03-19)

### Phase 1 results (9B judge — unreliable)

Scored by 9B judge (no thinking). All methods give identical scores because main == small model.

| ID | Status | Easy Baseline | Easy Ensemble | Easy Audit | Easy Hybrid | Hard Baseline | Hard Ensemble | Hard Audit | Hard Hybrid | Notes |
|----|:------:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|-------|
| v01 | [X] | 50.0% | 50.0% | 50.0% | 50.0% | 60.0% | 60.0% | 60.0% | 60.0% | H100, 7s/case |
| v02 | [!] | ~~100%~~ | ~~100%~~ | ~~100%~~ | ~~100%~~ | ~~100%~~ | ~~100%~~ | ~~100%~~ | ~~100%~~ | **INVALID** — thinking token leak: outputs are `"Output Generation:"` not diagnoses |
| v03 | [X] | 50.0% | 50.0% | 50.0% | 50.0% | 50.0% | 50.0% | 50.0% | 50.0% | 4bit, 10s/case |
| v04 | [X] | 60.0% | 60.0% | 60.0% | 60.0% | 60.0% | 60.0% | 60.0% | 60.0% | 8bit, 25s/case |
| v05 | [X] | 70.0% | 70.0% | 70.0% | 70.0% | 50.0% | 50.0% | 50.0% | 50.0% | H200, 27B-4bit main, 25s/case |

### Phase 1 rejudged by v06 (27B bf16 judge — reliable)

Same generation outputs as Phase 1, re-evaluated by 27B bf16 judge (semantic matching, not exact).

| ID | Status | Easy Baseline | Easy Ensemble | Easy Audit | Easy Hybrid | Hard Baseline | Hard Ensemble | Hard Audit | Hard Hybrid | Notes |
|----|:------:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|-------|
| v01 | [X] | 50.0% | 50.0% | 50.0% | 50.0% | 70.0% | 70.0% | 70.0% | 70.0% | Hard jumped 60→70% with better judge |
| v02 | [!] | 10.0% | 10.0% | 10.0% | 10.0% | 50.0% | 50.0% | 50.0% | 50.0% | Thinking leak garbled outputs; 27B judge salvaged some |
| v03 | [X] | 50.0% | 50.0% | 50.0% | 50.0% | 60.0% | 60.0% | 60.0% | 60.0% | Hard improved 50→60% |
| v04 | [X] | 60.0% | 60.0% | 60.0% | 60.0% | 70.0% | 70.0% | 70.0% | 70.0% | Hard jumped 60→70% |
| v05 | [X] | 70.0% | 70.0% | 70.0% | 70.0% | 60.0% | 60.0% | 60.0% | 60.0% | Hard improved 50→60% |

### Phase 2 results (27B bf16 judge — reliable)

| ID | Status | Easy Baseline | Easy Ensemble | Easy Audit | Easy Hybrid | Hard Baseline | Hard Ensemble | Hard Audit | Hard Hybrid | Notes |
|----|:------:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|-------|
| v06 | [X] | — | — | — | — | — | — | — | — | Rejudge only (no generation), see table above |
| v07 | [X] | 70.0% | 70.0% | 70.0% | 70.0% | 0.0% | 0.0% | 0.0% | 0.0% | Best Easy; Hard=0% (exact match, diagnoses close but not identical) |
| v08 | [R] | 10.0% | 10.0% | 10.0% | 10.0% | — | — | — | — | **Thinking leak again**: outputs are `"Construct Final Response:"`. Hard running. |
| v09 | [X] | 70.0% | 70.0% | 70.0% | 70.0% | 0.0% | 0.0% | 0.0% | 0.0% | True ensemble (27B+9B) — same Easy as v07, Hard=0% exact match |
| v10 | [!] | 20.0% | 20.0% | 20.0% | 20.0% | 10.0% | 10.0% | 10.0% | 10.0% | **Thinking leak**: outputs are `"Draft Output:"`. 9B+thinking broken. |

### Phase 3 results (pipeline fixes — not started)

| ID | Status | Easy Baseline | Easy Ensemble | Easy Audit | Easy Hybrid | Hard Baseline | Hard Ensemble | Hard Audit | Hard Hybrid | Notes |
|----|:------:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|-------|
| v11 | [ ] | — | — | — | — | — | — | — | — | Issue #1: thinking fix |
| v12 | [ ] | — | — | — | — | — | — | — | — | Issue #2: diverse candidates |
| v13 | [ ] | — | — | — | — | — | — | — | — | Issue #3: differential audit |

### Status key

- `[ ]` = not started
- `[X]` = done, results recorded
- `[!]` = failed / broken — see Notes column
- `[R]` = running

### Observations

**Phase 1:**
1. **v02 results are INVALID.** Thinking token leak: outputs are garbage strings like `"Output Generation:"`.
   Fixed in Phase 2 by separating `--enable-thinking` (generation) from `--judge-thinking` (judge).
2. **All 4 methods give identical scores** because main == small model in v01-v05.
   No ensemble diversity → no ensemble benefit. v09 fixes this with 27B main + 9B small.
3. **v06 rejudge shows 9B judge was too strict on Hard** — scores improved 10-20% with 27B judge
   (semantic matching catches near-miss diagnoses that exact match misses).
4. **27B-4bit (v05) best Easy at 70%**, 8bit (v04) best Hard at 70% (rejudged).

**Phase 2:**
5. **Thinking token leak is NOT fixed for v08/v10.** Despite the Phase 2 fix separating gen/judge
   thinking, the `hf_client.py` thinking extraction still fails — outputs like `"Construct Final
   Response:"` and `"Draft Output:"` leak through instead of actual diagnoses. **Phase 3 v11
   specifically targets this bug.**
6. **v07 and v09 match on Easy (70%)** but both score 0% Hard on exact match. The 27B judge in
   v06 rejudge scored similar configs at 60-70% Hard — the issue is exact-match scoring, not model
   quality. The actual diagnoses are close but not string-identical to gold labels.
7. **Ensemble (v09) shows no benefit over single-model (v07)** — identical scores. Candidate
   lists are still length-1 (no diversity). Phase 3 v12 addresses this.

## Phase 4: N=100 scale-up (pending v08/v10 resubmission results)

Top 4 configs from Phase 1-2, all with 27B bf16 judge, scaled to N=100 for statistical confidence.

| ID | Main | Thinking | Judge | Time limit | SLURM script | What it tests |
|----|------|:---:|-------|:---:|-------------|---------------|
| v07-n100 | 27B bf16 | off | 27B bf16 | 3h | `run_v07_n100.sh` | Best quality baseline |
| v08-n100 | 27B bf16 | **on** | 27B bf16 | 18h | `run_v08_n100.sh` | Does CoT help 27B? |
| v09-n100 | 27B bf16 + 9B bf16 | off | 27B bf16 | 3h | `run_v09_n100.sh` | True ensemble |
| v10-n100 | 9B bf16 | **on** | 27B bf16 | 18h | `run_v10_n100.sh` | Cheapest good config? |

**Prerequisite:** v08/v10 N=10 resubmissions (jobs 10685418/10685419) must finish successfully
with the thinking header fix (commit `214a9d8`) before launching N=100.

**Launch (after confirming fix works):**
```bash
cd /orcd/pool/005/sebasmos/code/CUPCase-humbleai
sbatch slurm-hf/run_v07_n100.sh
sbatch slurm-hf/run_v08_n100.sh
sbatch slurm-hf/run_v09_n100.sh
sbatch slurm-hf/run_v10_n100.sh
```

**Output:** `output-hf/vXX-n100-{description}/qwen35_{easy,hard}_100.csv`

## Launch

```bash
cd /orcd/pool/005/sebasmos/code/CUPCase-humbleai

# Phase 1-3 (N=10)
bash slurm-hf/launch_all.sh   # or individually: sbatch slurm-hf/run_v01.sh

# Phase 4 (N=100) — after v08/v10 fix confirmed
sbatch slurm-hf/run_v07_n100.sh
sbatch slurm-hf/run_v08_n100.sh
sbatch slurm-hf/run_v09_n100.sh
sbatch slurm-hf/run_v10_n100.sh

# Monitor
squeue -u $USER | grep hf-v
```

## Output

Each variant writes to `output-hf/vXX-{description}/`:
- `qwen35_easy_{N}.csv` — raw pipeline results on Easy dataset
- `qwen35_hard_{N}.csv` — raw pipeline results on Hard dataset
