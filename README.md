# CUPCase: Investigating Complex Disease Diagnosis With Large Language Models

This is the official repository for the CUPCase paper and dataset (AAAI 2025).

**Paper Link:** https://ojs.aaai.org/index.php/AAAI/article/view/35050

**Dataset:** https://huggingface.co/datasets/ofir408/CupCase

---

## Repository Structure

```
cupcase-humbleai/
├── lm_eval_evaluation/           # Evaluation framework for on-premise models
├── gpt_and_med_lm_evaluation/    # API-based model evaluation (GPT-4o, MedLM)
│   ├── premortem/                # Pre-Mortem Selective Analysis module
│   │   ├── config.py             # Configuration and thresholds
│   │   ├── quadrant_classifier.py # Risk quadrant classification
│   │   ├── premortem_prompts.py  # Prompt templates
│   │   └── belief_revision.py    # Belief revision engine
│   ├── evaluation_with_premortem.py  # Main evaluation script
│   ├── gpt_qa_eval.py            # Original MCQ evaluation
│   └── gpt_free_text_eval.py     # Original free-text evaluation
├── tests/                        # Test suite for Pre-Mortem module
├── utils/                        # Dataset utilities
└── preprocess/                   # Data preprocessing scripts
```

---

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/your-org/cupcase-humbleai.git
cd cupcase-humbleai

# Install dependencies
pip install -r gpt_and_med_lm_evaluation/requirements.txt

# Set up your OpenAI API key
export OPENAI_API_KEY="sk-your-api-key-here"
```

### 2. Prepare Data

Ensure you have the dataset file. You can either:
- Download from HuggingFace: https://huggingface.co/datasets/ofir408/CupCase
- Use the ablation study file with pre-computed token truncations

---

## Running Evaluations

### Standard Evaluation (Baseline - No Pre-Mortem)

For standard evaluation without the Pre-Mortem analysis:

```bash
# MCQ evaluation (baseline)
python gpt_and_med_lm_evaluation/evaluation_with_premortem.py \
    --task mcq \
    --no-premortem \
    --samples 250 \
    --batches 4

# Free-text evaluation (baseline)
python gpt_and_med_lm_evaluation/evaluation_with_premortem.py \
    --task free_text \
    --no-premortem \
    --samples 250 \
    --batches 4
```

### Evaluation with Pre-Mortem Analysis

The Pre-Mortem system reduces confirmation bias by:
1. Generating an initial hypothesis with 20% of the case
2. Classifying the case into a risk quadrant
3. Running Pre-Mortem analysis for high-risk cases
4. Generating final diagnosis with full case + belief revision

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

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--task` | Evaluation type: `mcq` or `free_text` | `mcq` |
| `--premortem` | Enable Pre-Mortem analysis | Disabled |
| `--no-premortem` | Disable Pre-Mortem (baseline) | - |
| `--samples` | Number of samples per batch | 250 |
| `--batches` | Number of batches to run | 4 |
| `--model` | Model to use | `gpt-4o` |
| `--complexity-threshold` | Threshold for complexity classification | 0.5 |
| `--stakes-threshold` | Threshold for stakes classification | 0.5 |
| `--output-dir` | Output directory for results | `output` |
| `--verbose` | Enable verbose output | Disabled |
| `--seed` | Random seed for reproducibility | 42 |

### Examples

```bash
# Quick test with verbose output
python gpt_and_med_lm_evaluation/evaluation_with_premortem.py \
    --task mcq \
    --premortem \
    --samples 10 \
    --batches 1 \
    --verbose

# Comparison study: run both baseline and Pre-Mortem
python gpt_and_med_lm_evaluation/evaluation_with_premortem.py \
    --task mcq --no-premortem --samples 100 --batches 2

python gpt_and_med_lm_evaluation/evaluation_with_premortem.py \
    --task mcq --premortem --samples 100 --batches 2

# Custom thresholds (more aggressive Pre-Mortem triggering)
python gpt_and_med_lm_evaluation/evaluation_with_premortem.py \
    --task mcq \
    --premortem \
    --complexity-threshold 0.3 \
    --stakes-threshold 0.3
```

---

## Pre-Mortem System Overview

### Risk Quadrant Classification

Cases are classified into 4 quadrants based on clinical complexity and stakes:

| Quadrant | Complexity | Stakes | Pre-Mortem |
|----------|------------|--------|------------|
| **Q1 (Routine)** | Low | Low | No |
| **Q2 (Watchful)** | Low | High | Yes |
| **Q3 (Curiosity)** | High | Low | Optional |
| **Q4 (Escalate)** | High | High | Yes |

### Red Flag Detection

Certain clinical patterns always trigger Pre-Mortem regardless of quadrant:
- Chest pain, shortness of breath, syncope
- Altered mental status, sudden onset symptoms
- Immunocompromised patients
- And more...

### Pipeline Flow

```
Case Input (20% tokens)
        │
        ▼
┌───────────────────┐
│ Pass 1: Generate  │
│ Initial Hypothesis│
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ Quadrant          │
│ Classification    │
└─────────┬─────────┘
          │
    ┌─────┴─────┐
    │           │
    ▼           ▼
[Low Risk]  [High Risk]
    │           │
    │           ▼
    │   ┌───────────────┐
    │   │ Pre-Mortem:   │
    │   │ "What if I'm  │
    │   │  wrong?"      │
    │   └───────┬───────┘
    │           │
    └─────┬─────┘
          │
          ▼
┌───────────────────┐
│ Pass 2: Final     │
│ Diagnosis (100%)  │
└───────────────────┘
```

---

## Running Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest tests/ -v

# Run specific test files
pytest tests/test_quadrant_classifier.py -v
pytest tests/test_premortem_prompts.py -v
pytest tests/test_belief_revision.py -v
pytest tests/test_integration.py -v

# Run with coverage report
pytest tests/ --cov=gpt_and_med_lm_evaluation/premortem --cov-report=html

# Quick smoke test (no pytest required)
python3 -c "
import sys
sys.path.insert(0, 'gpt_and_med_lm_evaluation')
from premortem import PreMortemConfig, QuadrantClassifier
classifier = QuadrantClassifier()
result = classifier.classify('Patient with chest pain and shortness of breath')
print(f'Quadrant: {result.quadrant.name}')
print(f'Red flags: {result.red_flags_detected}')
print(f'Requires Pre-Mortem: {result.requires_premortem}')
print('All imports successful!')
"
```

---

## Output Files

Results are saved to the `output/` directory:

- `{task}_{mode}_{timestamp}.csv` - Detailed results for each case
- `{task}_{mode}_{timestamp}_metrics.json` - Summary metrics

### Metrics Included

**For MCQ:**
- Overall accuracy
- Accuracy with/without Pre-Mortem
- Accuracy by quadrant
- Belief revision rate

**For Free-Text:**
- BERTScore (Precision, Recall, F1)
- Pre-Mortem application rate

---

## Using the Module Programmatically

```python
from openai import OpenAI
import sys
sys.path.insert(0, 'gpt_and_med_lm_evaluation')

from premortem import (
    PreMortemConfig,
    BeliefRevisionEngine,
    QuadrantClassifier
)

# Initialize
client = OpenAI(api_key="your-key")
config = PreMortemConfig(
    enable_premortem=True,
    complexity_threshold=0.5,
    stakes_threshold=0.5
)

# Option 1: Use the full engine
engine = BeliefRevisionEngine(client, config)
result = engine.evaluate_case(
    case_text_20pct="Patient with chest pain...",
    case_text_full="Full case presentation...",
    task_type="free_text"
)
print(f"Final diagnosis: {result.final_diagnosis}")
print(f"Pre-Mortem applied: {result.premortem_applied}")

# Option 2: Just classify risk
classifier = QuadrantClassifier()
quadrant_result = classifier.classify("Patient case text...")
print(f"Quadrant: {quadrant_result.quadrant.name}")
print(f"Requires Pre-Mortem: {quadrant_result.requires_premortem}")
```

---

## Citation

If you use CupCase or find this repository useful for your research or work, please cite us using the following citation:

```bibtex
@inproceedings{perets2025cupcase,
  title={CUPCase: Clinically Uncommon Patient Cases and Diagnoses Dataset},
  author={Perets, Oriel and Shoham, Ofir Ben and Grinberg, Nir and Rappoport, Nadav},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={27},
  pages={28293--28301},
  year={2025}
}
```

---

## License

See LICENSE file for details.
