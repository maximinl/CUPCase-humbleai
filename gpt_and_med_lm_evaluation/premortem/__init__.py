"""
Pre-Mortem Selective Analysis Module.

This module implements the Pre-Mortem Selective system with Risk Quadrant Trigger
for reducing confirmation bias in medical diagnosis.

Components:
- config: Configuration and thresholds for the system
- quadrant_classifier: Risk quadrant classification based on complexity and stakes
- premortem_prompts: Prompt templates for Pre-Mortem analysis
- belief_revision: Belief revision engine integrating all components
"""

from .config import PreMortemConfig, RiskQuadrant
from .quadrant_classifier import QuadrantClassifier, QuadrantResult
from .premortem_prompts import PreMortemPrompts, PreMortemResponse
from .belief_revision import BeliefRevisionEngine, DiagnosisResult

__all__ = [
    "PreMortemConfig",
    "RiskQuadrant",
    "QuadrantClassifier",
    "QuadrantResult",
    "PreMortemPrompts",
    "PreMortemResponse",
    "BeliefRevisionEngine",
    "DiagnosisResult",
]

__version__ = "1.0.0"
