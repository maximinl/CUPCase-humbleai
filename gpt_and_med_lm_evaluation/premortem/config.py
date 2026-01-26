"""
Configuration for Pre-Mortem Selective Analysis.

Defines thresholds, parameters, and constants for the Pre-Mortem system
including risk quadrant definitions and clinical indicator keywords.
"""

from dataclasses import dataclass, field
from typing import List
from enum import Enum


class RiskQuadrant(Enum):
    """
    Cuadrantes del Dual-Reflective Trigger Map.

    Based on the intersection of clinical complexity and clinical stakes:
    - ROUTINE (Q1): Low complexity, low stakes -> No Pre-Mortem needed
    - WATCHFUL (Q2): Low complexity, high stakes -> Pre-Mortem required
    - CURIOSITY (Q3): High complexity, low stakes -> Pre-Mortem optional
    - ESCALATE (Q4): High complexity, high stakes -> Pre-Mortem mandatory
    """
    ROUTINE = 1      # Baja complejidad, bajo stake -> Sin Pre-Mortem
    WATCHFUL = 2     # Baja complejidad, alto stake -> Pre-Mortem
    CURIOSITY = 3    # Alta complejidad, bajo stake -> Pre-Mortem opcional
    ESCALATE = 4     # Alta complejidad, alto stake -> Pre-Mortem obligatorio


@dataclass
class PreMortemConfig:
    """
    Configuration for the Pre-Mortem system.

    Attributes:
        complexity_threshold: Threshold for classifying clinical complexity (0-1)
        stakes_threshold: Threshold for classifying clinical stakes (0-1)
        premortem_quadrants: List of quadrants that trigger Pre-Mortem analysis
        evidence_strength_threshold: If alternative evidence > threshold, escalate
        initial_token_percentage: Percentage of tokens for Pass 1 (default 20%)
        model_name: LLM model to use for inference
        temperature: Temperature for LLM generation
        enable_premortem: Global flag to enable/disable Pre-Mortem
        verbose: Enable verbose logging
    """

    # Thresholds para clasificacion de cuadrantes
    complexity_threshold: float = 0.5
    stakes_threshold: float = 0.5

    # Thresholds para trigger de Pre-Mortem
    premortem_quadrants: List[RiskQuadrant] = field(default_factory=list)

    # Threshold para evidencia del Pre-Mortem
    evidence_strength_threshold: float = 0.6

    # Configuracion de tokens progresivos
    initial_token_percentage: float = 0.2

    # Modelo a usar
    model_name: str = "gpt-4o"
    temperature: float = 0.0

    # Flags de control
    enable_premortem: bool = True
    verbose: bool = False

    def __post_init__(self):
        """Initialize default premortem quadrants if not provided."""
        if not self.premortem_quadrants:
            # Por defecto, Pre-Mortem en cuadrantes de alto riesgo
            self.premortem_quadrants = [
                RiskQuadrant.WATCHFUL,
                RiskQuadrant.ESCALATE
            ]


# Palabras clave para deteccion de complejidad clinica
COMPLEXITY_INDICATORS = {
    "high": [
        "multiple", "comorbid", "atypical", "rare", "unusual", "complex",
        "unclear", "uncertain", "conflicting", "inconsistent", "progressive",
        "refractory", "resistant", "recurrent", "chronic", "acute on chronic",
        "multifactorial", "idiopathic", "cryptogenic", "undifferentiated",
        "multisystem", "overlap", "mixed", "heterogeneous", "variable"
    ],
    "low": [
        "typical", "classic", "straightforward", "common", "mild",
        "stable", "resolved", "improving", "uncomplicated", "simple",
        "isolated", "localized", "single", "clear", "definite"
    ]
}

# Palabras clave para deteccion de stakes clinicos
STAKES_INDICATORS = {
    "high": [
        "emergency", "urgent", "critical", "life-threatening", "sepsis",
        "shock", "arrest", "hemorrhage", "stroke", "infarction", "malignant",
        "metastatic", "fulminant", "acute", "severe", "unstable", "deteriorating",
        "respiratory failure", "cardiac", "renal failure", "hepatic failure",
        "coma", "unresponsive", "intubated", "icu", "intensive care",
        "disseminated", "massive", "profound", "crisis"
    ],
    "low": [
        "benign", "self-limiting", "minor", "routine", "follow-up",
        "outpatient", "elective", "stable", "controlled", "managed",
        "chronic stable", "maintenance", "surveillance", "screening"
    ]
}

# Red flags que siempre activan Pre-Mortem independientemente del cuadrante
RED_FLAG_PATTERNS = [
    "chest pain",
    "shortness of breath",
    "syncope",
    "altered mental status",
    "sudden onset",
    "worst headache",
    "focal neurological",
    "hemodynamic",
    "immunocompromised",
    "pregnancy",
    "pediatric",
    "elderly",
    "fever and rash",
    "neck stiffness",
    "photophobia",
    "altered consciousness",
    "seizure",
    "weight loss",
    "night sweats",
    "hemoptysis",
    "melena",
    "hematochezia",
    "jaundice",
    "acute abdomen",
    "testicular pain",
    "vision loss",
    "limb weakness"
]


# Mapping of quadrant names to descriptions for logging
QUADRANT_DESCRIPTIONS = {
    RiskQuadrant.ROUTINE: "Low complexity, low stakes - routine case",
    RiskQuadrant.WATCHFUL: "Low complexity, high stakes - needs vigilance",
    RiskQuadrant.CURIOSITY: "High complexity, low stakes - intellectually challenging",
    RiskQuadrant.ESCALATE: "High complexity, high stakes - maximum attention required"
}
