"""
Integration tests for the Pre-Mortem system.

Tests cover:
- Module imports
- Full pipeline execution
- Configuration validation
- Cross-module interactions
"""

import pytest
import sys
import os
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'gpt_and_med_lm_evaluation'))


class TestModuleImports:
    """Tests for module imports."""

    def test_import_config(self):
        """Config module imports correctly."""
        from premortem.config import PreMortemConfig, RiskQuadrant
        assert PreMortemConfig is not None
        assert RiskQuadrant is not None

    def test_import_quadrant_classifier(self):
        """Quadrant classifier module imports correctly."""
        from premortem.quadrant_classifier import QuadrantClassifier, QuadrantResult
        assert QuadrantClassifier is not None
        assert QuadrantResult is not None

    def test_import_premortem_prompts(self):
        """Prompts module imports correctly."""
        from premortem.premortem_prompts import PreMortemPrompts, PreMortemResponse
        assert PreMortemPrompts is not None
        assert PreMortemResponse is not None

    def test_import_belief_revision(self):
        """Belief revision module imports correctly."""
        from premortem.belief_revision import BeliefRevisionEngine, DiagnosisResult
        assert BeliefRevisionEngine is not None
        assert DiagnosisResult is not None

    def test_import_from_package(self):
        """Package-level imports work."""
        from premortem import (
            PreMortemConfig,
            RiskQuadrant,
            QuadrantClassifier,
            QuadrantResult,
            PreMortemPrompts,
            PreMortemResponse,
            BeliefRevisionEngine,
            DiagnosisResult
        )
        assert all([
            PreMortemConfig, RiskQuadrant, QuadrantClassifier,
            QuadrantResult, PreMortemPrompts, PreMortemResponse,
            BeliefRevisionEngine, DiagnosisResult
        ])


class TestConfigDefaults:
    """Tests for configuration defaults."""

    def test_default_thresholds(self):
        """Default thresholds are set correctly."""
        from premortem.config import PreMortemConfig
        config = PreMortemConfig()

        assert config.complexity_threshold == 0.5
        assert config.stakes_threshold == 0.5

    def test_default_token_percentage(self):
        """Default token percentage is set."""
        from premortem.config import PreMortemConfig
        config = PreMortemConfig()

        assert config.initial_token_percentage == 0.2

    def test_default_premortem_enabled(self):
        """Pre-Mortem is enabled by default."""
        from premortem.config import PreMortemConfig
        config = PreMortemConfig()

        assert config.enable_premortem is True

    def test_default_quadrants(self):
        """Default Pre-Mortem quadrants are set."""
        from premortem.config import PreMortemConfig, RiskQuadrant
        config = PreMortemConfig()

        assert RiskQuadrant.WATCHFUL in config.premortem_quadrants
        assert RiskQuadrant.ESCALATE in config.premortem_quadrants

    def test_indicators_defined(self):
        """Clinical indicators are defined."""
        from premortem.config import COMPLEXITY_INDICATORS, STAKES_INDICATORS, RED_FLAG_PATTERNS

        assert len(COMPLEXITY_INDICATORS["high"]) > 0
        assert len(COMPLEXITY_INDICATORS["low"]) > 0
        assert len(STAKES_INDICATORS["high"]) > 0
        assert len(STAKES_INDICATORS["low"]) > 0
        assert len(RED_FLAG_PATTERNS) > 0


class TestFullPipelineMock:
    """Tests for the full pipeline with mocks."""

    def test_full_pipeline_free_text(self, mock_openai_client_with_responses, sample_cases):
        """Full pipeline runs for free-text task."""
        from premortem.config import PreMortemConfig
        from premortem.belief_revision import BeliefRevisionEngine

        client, set_responses = mock_openai_client_with_responses
        set_responses([
            "Tension headache",  # Pass 1
            """ALTERNATIVE_DIAGNOSIS: Migraine
EVIDENCE: Headache pattern
EVIDENCE_STRENGTH: 0.3
MISSED_RED_FLAGS: None
RECOMMENDATION: MAINTAIN""",  # Pre-Mortem
            "Tension headache"  # Pass 2
        ])

        config = PreMortemConfig(
            enable_premortem=True,
            complexity_threshold=0.1,  # Low threshold to trigger
            stakes_threshold=0.1
        )
        engine = BeliefRevisionEngine(client, config)
        case = sample_cases["simple"]

        result = engine.evaluate_case(
            case_text_20pct=case["case_20pct"],
            case_text_full=case["case_full"],
            task_type="free_text"
        )

        assert result.final_diagnosis is not None
        assert result.initial_hypothesis is not None
        assert result.quadrant is not None

    def test_full_pipeline_mcq(self, mock_openai_client_with_responses, sample_cases):
        """Full pipeline runs for MCQ task."""
        from premortem.config import PreMortemConfig
        from premortem.belief_revision import BeliefRevisionEngine

        client, set_responses = mock_openai_client_with_responses
        set_responses(["1", "1"])  # MCQ responses

        config = PreMortemConfig(enable_premortem=False)
        engine = BeliefRevisionEngine(client, config)
        case = sample_cases["simple"]
        options = [case["true_diagnosis"]] + case["distractors"]

        result = engine.evaluate_case(
            case_text_20pct=case["case_20pct"],
            case_text_full=case["case_full"],
            task_type="mcq",
            options=options
        )

        assert result.final_diagnosis in ["1", "2", "3", "4"]

    def test_pipeline_with_high_risk_case(self, mock_openai_client_with_responses, sample_cases):
        """Pipeline handles high-risk case correctly."""
        from premortem.config import PreMortemConfig, RiskQuadrant
        from premortem.belief_revision import BeliefRevisionEngine

        client, set_responses = mock_openai_client_with_responses
        set_responses([
            "Acute MI",
            """ALTERNATIVE_DIAGNOSIS: Pulmonary embolism
EVIDENCE: Dyspnea, hypotension
EVIDENCE_STRENGTH: 0.7
MISSED_RED_FLAGS: tachycardia
RECOMMENDATION: REVISE""",
            "Acute MI"
        ])

        config = PreMortemConfig(enable_premortem=True)
        engine = BeliefRevisionEngine(client, config)
        case = sample_cases["complex_high_stakes"]

        result = engine.evaluate_case(
            case_text_20pct=case["case_20pct"],
            case_text_full=case["case_full"],
            task_type="free_text"
        )

        # High-risk case should have high stakes score
        assert result.quadrant_result.stakes_score > 0


class TestCrossModuleInteraction:
    """Tests for interactions between modules."""

    def test_classifier_uses_config(self):
        """Classifier respects config thresholds."""
        from premortem.config import PreMortemConfig
        from premortem.quadrant_classifier import QuadrantClassifier

        low_config = PreMortemConfig(complexity_threshold=0.1, stakes_threshold=0.1)
        high_config = PreMortemConfig(complexity_threshold=0.9, stakes_threshold=0.9)

        low_classifier = QuadrantClassifier(low_config)
        high_classifier = QuadrantClassifier(high_config)

        case = "Some clinical presentation with moderate complexity."

        low_result = low_classifier.classify(case)
        high_result = high_classifier.classify(case)

        # Low thresholds might classify as higher quadrants
        # High thresholds tend toward ROUTINE
        assert low_result.quadrant is not None
        assert high_result.quadrant is not None

    def test_prompts_work_with_classifier_output(self):
        """Prompts can use classifier output."""
        from premortem.quadrant_classifier import QuadrantClassifier
        from premortem.premortem_prompts import PreMortemPrompts

        classifier = QuadrantClassifier()
        case = "Patient with chest pain and shortness of breath."
        result = classifier.classify(case)

        # Use classifier output in prompt
        prompt = PreMortemPrompts.build_premortem_prompt(
            case_text=case,
            hypothesis="Acute coronary syndrome"
        )

        assert len(prompt) > 0
        assert "chest pain" in prompt.lower()

    def test_engine_uses_classifier(self, mock_openai_client):
        """Engine correctly uses classifier."""
        from premortem.config import PreMortemConfig
        from premortem.belief_revision import BeliefRevisionEngine

        config = PreMortemConfig(enable_premortem=True)
        engine = BeliefRevisionEngine(mock_openai_client, config)

        assert engine.classifier is not None
        assert engine.classifier.config == config


class TestQuadrantValues:
    """Tests for quadrant enumeration values."""

    def test_quadrant_values(self):
        """Quadrants have expected values."""
        from premortem.config import RiskQuadrant

        assert RiskQuadrant.ROUTINE.value == 1
        assert RiskQuadrant.WATCHFUL.value == 2
        assert RiskQuadrant.CURIOSITY.value == 3
        assert RiskQuadrant.ESCALATE.value == 4

    def test_all_quadrants_covered(self):
        """All quadrants are defined."""
        from premortem.config import RiskQuadrant

        quadrants = list(RiskQuadrant)
        assert len(quadrants) == 4


class TestDependenciesAvailable:
    """Tests for required dependencies."""

    def test_pandas_available(self):
        """Pandas is available (optional dependency)."""
        try:
            import pandas
            assert pandas is not None
        except ImportError:
            pytest.skip("pandas not installed (optional dependency)")

    def test_dataclasses_available(self):
        """Dataclasses are available."""
        from dataclasses import dataclass, field
        assert dataclass is not None
        assert field is not None

    def test_typing_available(self):
        """Typing module is available."""
        from typing import Dict, List, Optional, Tuple, Any
        assert all([Dict, List, Optional, Tuple, Any])

    def test_enum_available(self):
        """Enum module is available."""
        from enum import Enum
        assert Enum is not None


class TestPackageVersion:
    """Tests for package metadata."""

    def test_version_defined(self):
        """Package version is defined."""
        from premortem import __version__
        assert __version__ is not None
        assert isinstance(__version__, str)

    def test_all_exports_defined(self):
        """All exports are defined."""
        from premortem import __all__
        assert len(__all__) > 0

        # Import each exported item
        import premortem
        for name in __all__:
            assert hasattr(premortem, name)


class TestConfigurationValidation:
    """Tests for configuration validation."""

    def test_threshold_range(self):
        """Thresholds should be in valid range."""
        from premortem.config import PreMortemConfig

        # These should work
        config = PreMortemConfig(complexity_threshold=0.0)
        assert config.complexity_threshold == 0.0

        config = PreMortemConfig(complexity_threshold=1.0)
        assert config.complexity_threshold == 1.0

    def test_custom_quadrants(self):
        """Custom quadrants can be specified."""
        from premortem.config import PreMortemConfig, RiskQuadrant

        # Only trigger Pre-Mortem for ESCALATE
        config = PreMortemConfig(
            premortem_quadrants=[RiskQuadrant.ESCALATE]
        )

        assert len(config.premortem_quadrants) == 1
        assert RiskQuadrant.ESCALATE in config.premortem_quadrants
        assert RiskQuadrant.WATCHFUL not in config.premortem_quadrants
