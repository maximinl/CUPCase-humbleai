# CUPCase Diagnostic Pipeline (Internal Research)

This repository contains scripts and data for evaluating Large Language Models on the CUPCase Hard Test (Open-Ended Clinical Diagnosis). The project follows a three-stage evolutionary path to optimize diagnostic accuracy on real-world uncommon patient cases.

## Experimental Evolution

### Stage 1: Knowledge Ensemble (ensemble_v2.py)
Concept: Utilize multi-model consensus to maximize the retrieval of rare diagnoses (Zebras).
- Mechanism: Aggregates independent candidate lists from GPT-4o and DeepSeek-V3.
- Goal: Increase recall for low-frequency medical conditions.

### Stage 2: Systematic Reasoning Audit (audit_pipeline.py)
Concept: Implement a "System 2" reasoning layer to filter candidates based on clinical evidence.
- Mechanism: Forces a structured For vs. Against evidence check for each candidate diagnosis before selection.
- Goal: Mitigate anchor bias and logical shortcuts in standard zero-shot generation.

### Stage 3: Hybrid Pipeline (hybrid_boss_turbo.py)
Concept: Combine the high-recall retrieval of the Ensemble with the rigorous filtering of the Audit.
- Mechanism: Candidates generated in Stage 1 are passed through the Audit logic of Stage 2.
- Performance: Stabilized at 54.6% Accuracy on N=350 samples, significantly outperforming the unassisted GPT-4o baseline (~42%).

## Execution Commands

Stage 1:
  python ensemble_v2.py --samples 350 --data-path datasets/Case_report_w_images_dis_VF.csv

Stage 2:
  python audit_pipeline.py --samples 350 --data-path datasets/Case_report_w_images_dis_VF.csv

Stage 3:
  python hybrid_boss_turbo.py --samples 350 --data-path datasets/Case_report_w_images_dis_VF.csv

## Results Summary

Logs for the consolidated N=350 run are located in:
  /results/turbo_results_350.csv
