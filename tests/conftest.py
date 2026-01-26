"""
Pytest fixtures and configuration for Pre-Mortem tests.

Provides reusable test fixtures including sample cases, mock clients,
and configuration objects.
"""

import sys
import os
import pytest
from unittest.mock import MagicMock

# Add the parent directory to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'gpt_and_med_lm_evaluation'))

from premortem.config import PreMortemConfig, RiskQuadrant


# ============================================================================
# Sample clinical cases for testing
# ============================================================================

SAMPLE_CASES = {
    "simple": {
        "case_20pct": "A 45-year-old male presents with mild headache for 2 days.",
        "case_full": (
            "A 45-year-old male presents with mild headache for 2 days. "
            "No fever, no neck stiffness. Normal neurological exam. "
            "History of tension headaches. Takes ibuprofen with relief. "
            "No visual changes, no nausea or vomiting."
        ),
        "true_diagnosis": "Tension headache",
        "distractors": ["Migraine", "Meningitis", "Brain tumor"]
    },
    "complex_high_stakes": {
        "case_20pct": (
            "A 65-year-old female presents with sudden onset severe chest pain "
            "and shortness of breath."
        ),
        "case_full": (
            "A 65-year-old female presents with sudden onset severe chest pain "
            "and shortness of breath. She is diaphoretic and hypotensive. "
            "ECG shows ST elevation in leads V1-V4. Troponin pending. "
            "History of hypertension and diabetes. Previous smoker. "
            "Pain radiating to left arm. BP 90/60, HR 110, RR 24."
        ),
        "true_diagnosis": "Acute myocardial infarction",
        "distractors": ["Pulmonary embolism", "Aortic dissection", "Pneumothorax"]
    },
    "complex_low_stakes": {
        "case_20pct": (
            "A 30-year-old female presents with multiple joint pains and fatigue."
        ),
        "case_full": (
            "A 30-year-old female presents with multiple joint pains, fatigue, "
            "and a butterfly rash on her face. ANA positive, anti-dsDNA positive. "
            "CBC shows mild anemia and lymphopenia. Urinalysis shows proteinuria. "
            "Symptoms have been progressive over the past 6 months. "
            "No acute distress, vital signs stable."
        ),
        "true_diagnosis": "Systemic lupus erythematosus",
        "distractors": ["Rheumatoid arthritis", "Fibromyalgia", "Viral syndrome"]
    },
    "low_complexity_high_stakes": {
        "case_20pct": (
            "A 50-year-old male presents to the emergency room with severe abdominal pain."
        ),
        "case_full": (
            "A 50-year-old male presents to the emergency room with severe abdominal pain. "
            "The pain is sudden onset, diffuse, and associated with vomiting. "
            "Abdomen is rigid and tender. Patient appears septic with fever 39C. "
            "CT shows free air under the diaphragm."
        ),
        "true_diagnosis": "Perforated viscus",
        "distractors": ["Acute appendicitis", "Pancreatitis", "Bowel obstruction"]
    }
}


# ============================================================================
# Pytest fixtures
# ============================================================================

@pytest.fixture
def sample_cases():
    """Provide sample clinical cases for testing."""
    return SAMPLE_CASES


@pytest.fixture
def simple_case():
    """Provide a simple (routine) case."""
    return SAMPLE_CASES["simple"]


@pytest.fixture
def high_risk_case():
    """Provide a high-risk case."""
    return SAMPLE_CASES["complex_high_stakes"]


@pytest.fixture
def mock_openai_client():
    """
    Create a mock OpenAI client for testing.

    The mock returns configurable responses for the chat completions API.
    """
    client = MagicMock()

    def mock_completion(*args, **kwargs):
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = "1"  # Default MCQ response
        return response

    client.chat.completions.create = mock_completion
    return client


@pytest.fixture
def mock_openai_client_with_responses():
    """
    Create a mock OpenAI client that cycles through predefined responses.

    Usage:
        def test_something(mock_openai_client_with_responses):
            client, set_responses = mock_openai_client_with_responses
            set_responses(["response1", "response2", "response3"])
            # Client will return these responses in order
    """
    client = MagicMock()
    responses = []
    call_count = [0]

    def set_responses(new_responses):
        nonlocal responses
        responses[:] = new_responses
        call_count[0] = 0

    def mock_completion(*args, **kwargs):
        response = MagicMock()
        response.choices = [MagicMock()]
        idx = min(call_count[0], len(responses) - 1) if responses else 0
        response.choices[0].message.content = responses[idx] if responses else "1"
        call_count[0] += 1
        return response

    client.chat.completions.create = mock_completion
    return client, set_responses


@pytest.fixture
def default_config():
    """Provide default PreMortemConfig."""
    return PreMortemConfig()


@pytest.fixture
def premortem_enabled_config():
    """Provide PreMortemConfig with Pre-Mortem enabled."""
    return PreMortemConfig(
        enable_premortem=True,
        complexity_threshold=0.5,
        stakes_threshold=0.5,
        verbose=False
    )


@pytest.fixture
def premortem_disabled_config():
    """Provide PreMortemConfig with Pre-Mortem disabled."""
    return PreMortemConfig(
        enable_premortem=False,
        verbose=False
    )


@pytest.fixture
def low_threshold_config():
    """Provide config with low thresholds (triggers Pre-Mortem more often)."""
    return PreMortemConfig(
        enable_premortem=True,
        complexity_threshold=0.3,
        stakes_threshold=0.3,
        verbose=False
    )


@pytest.fixture
def high_threshold_config():
    """Provide config with high thresholds (triggers Pre-Mortem less often)."""
    return PreMortemConfig(
        enable_premortem=True,
        complexity_threshold=0.7,
        stakes_threshold=0.7,
        verbose=False
    )


# ============================================================================
# Test case generators
# ============================================================================

@pytest.fixture
def case_with_red_flags():
    """Provide a case text with multiple red flags."""
    return (
        "A 70-year-old elderly patient presents with chest pain and "
        "shortness of breath. She reports syncope earlier today and "
        "has altered mental status. History of immunocompromised state."
    )


@pytest.fixture
def case_without_red_flags():
    """Provide a case text without red flags."""
    return (
        "A 35-year-old healthy male presents with a minor skin rash "
        "on his forearm. The rash appeared 3 days ago after gardening. "
        "No systemic symptoms. Vital signs are normal."
    )


@pytest.fixture
def premortem_response_text():
    """Provide a sample Pre-Mortem response text for parsing tests."""
    return """ALTERNATIVE_DIAGNOSIS: Pulmonary embolism
EVIDENCE: Sudden onset dyspnea, tachycardia, recent immobility
EVIDENCE_STRENGTH: 0.75
MISSED_RED_FLAGS: tachycardia, hypoxia, unilateral leg swelling
RECOMMENDATION: REVISE"""


@pytest.fixture
def premortem_response_incomplete():
    """Provide an incomplete Pre-Mortem response for parsing edge cases."""
    return """ALTERNATIVE_DIAGNOSIS: Some diagnosis
EVIDENCE_STRENGTH: invalid
RECOMMENDATION: ESCALATE"""
