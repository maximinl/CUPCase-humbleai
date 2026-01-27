# CUPCase-HumbleAI: Pre-Mortem Selective Analysis for Medical Diagnosis

> **Forked from:** [ofir408/CUPCase](https://github.com/ofir408/CUPCase) - Original CUPCase paper and dataset (AAAI 2025)

This repository extends CUPCase with a **Pre-Mortem Selective Analysis** system to reduce confirmation bias in LLM-based medical diagnosis.

**Dataset:** https://huggingface.co/datasets/ofir408/CupCase

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/sebasmos/CUPCase-humbleai.git
cd CUPCase-humbleai

# Install dependencies
pip install -r gpt_and_med_lm_evaluation/requirements.txt

# Set up your OpenAI API key
export OPENAI_API_KEY="sk-your-api-key-here"
```

---

## Running Evaluations

### Baseline (Standard Evaluation)

```bash
# MCQ evaluation - baseline
python gpt_and_med_lm_evaluation/evaluation_with_premortem.py \
    --task mcq \
    --no-premortem \
    --samples 250 \
    --batches 4

# Free-text evaluation - baseline
python gpt_and_med_lm_evaluation/evaluation_with_premortem.py \
    --task free_text \
    --no-premortem \
    --samples 250 \
    --batches 4
```

### With Pre-Mortem Analysis

The Pre-Mortem system forces the model to challenge its initial hypothesis before final diagnosis:

```bash
# MCQ evaluation with Pre-Mortem
python gpt_and_med_lm_evaluation/evaluation_with_premortem.py \
    --task mcq \
    --premortem \
    --samples 250 \
    --batches 4

# Free-text evaluation with Pre-Mortem
python gpt_and_med_lm_evaluation/evaluation_with_premortem.py \
    --task free_text \
    --premortem \
    --samples 250 \
    --batches 4
```

### Quick Test

```bash
# Quick test with 10 samples
python gpt_and_med_lm_evaluation/evaluation_with_premortem.py \
    --task mcq \
    --premortem \
    --samples 10 \
    --batches 1 \
    --verbose
```

---

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--task` | `mcq` or `free_text` | `mcq` |
| `--premortem` | Enable Pre-Mortem analysis | Disabled |
| `--no-premortem` | Baseline mode (no Pre-Mortem) | - |
| `--samples` | Samples per batch | 250 |
| `--batches` | Number of batches | 4 |
| `--model` | Model to use | `gpt-4o` |
| `--complexity-threshold` | Complexity threshold | 0.5 |
| `--stakes-threshold` | Stakes threshold | 0.5 |
| `--output-dir` | Output directory | `output` |
| `--verbose` | Verbose output | Disabled |

---

## Pre-Mortem System

### How It Works

1. **Pass 1**: Generate initial hypothesis using 20% of case tokens
2. **Risk Classification**: Classify case into risk quadrant
3. **Pre-Mortem** (high-risk only): Challenge hypothesis - "What dangerous condition might I be missing?"
4. **Pass 2**: Final diagnosis with full case + belief revision

### Risk Quadrants

| Quadrant | Complexity | Stakes | Pre-Mortem |
|----------|------------|--------|------------|
| Q1 (Routine) | Low | Low | No |
| Q2 (Watchful) | Low | High | Yes |
| Q3 (Curiosity) | High | Low | Optional |
| Q4 (Escalate) | High | High | Yes |

### Pipeline

```
Case (20% tokens) → Pass 1 → Quadrant Classification
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
               [Low Risk]                      [High Risk]
                    │                               │
                    │                    Pre-Mortem Analysis
                    │                    "What if I'm wrong?"
                    │                               │
                    └───────────────┬───────────────┘
                                    ▼
                         Pass 2: Final Diagnosis
                            (100% tokens)
```

---

## Running Tests

```bash
# Install pytest
pip install pytest

# Run all tests
pytest tests/ -v

# Quick smoke test (no pytest needed)
python3 -c "
import sys
sys.path.insert(0, 'gpt_and_med_lm_evaluation')
from premortem import QuadrantClassifier
classifier = QuadrantClassifier()
result = classifier.classify('Patient with chest pain and shortness of breath')
print(f'Quadrant: {result.quadrant.name}')
print(f'Requires Pre-Mortem: {result.requires_premortem}')
"
```

---

## Output

Results saved to `output/`:
- `{task}_{mode}_{timestamp}.csv` - Detailed results
- `{task}_{mode}_{timestamp}_metrics.json` - Summary metrics (accuracy, BERTScore, etc.)

---

## Repository Structure

```
CUPCase-humbleai/
├── gpt_and_med_lm_evaluation/
│   ├── premortem/                    # Pre-Mortem module
│   │   ├── config.py                 # Configuration
│   │   ├── quadrant_classifier.py    # Risk classification
│   │   ├── premortem_prompts.py      # Prompts
│   │   └── belief_revision.py        # Engine
│   └── evaluation_with_premortem.py  # Main script
├── tests/                            # Test suite (102 tests)
├── lm_eval_evaluation/               # On-premise model evaluation
├── utils/                            # Utilities
└── preprocess/                       # Preprocessing
```

---

## Original Work

This repository is forked from [ofir408/CUPCase](https://github.com/ofir408/CUPCase).

For the original CUPCase paper and citation, please refer to the original repository.
