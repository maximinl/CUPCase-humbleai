"""
Tests for the BeliefRevisionEngine module.

Tests cover:
- Engine initialization
- Case evaluation pipeline
- Pre-Mortem triggering
- Belief revision detection
- Diagnosis result structure
"""

import pytest
import sys
import os
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'gpt_and_med_lm_evaluation'))

from premortem.belief_revision import (
    BeliefRevisionEngine,
    DiagnosisResult,
    evaluate_case_with_premortem
)
from premortem.config import PreMortemConfig, RiskQuadrant
from premortem.premortem_prompts import PreMortemResponse


class TestBeliefRevisionEngineInit:
    """Tests for engine initialization."""

    def test_init_with_default_config(self, mock_openai_client):
        """Engine initializes with default config."""
        engine = BeliefRevisionEngine(mock_openai_client)
        assert engine.config is not None
        assert engine.client == mock_openai_client
        assert engine.classifier is not None

    def test_init_with_custom_config(self, mock_openai_client, premortem_enabled_config):
        """Engine initializes with custom config."""
        engine = BeliefRevisionEngine(mock_openai_client, premortem_enabled_config)
        assert engine.config == premortem_enabled_config


class TestEvaluateCase:
    """Tests for the evaluate_case method."""

    def test_returns_diagnosis_result(
        self, mock_openai_client, premortem_enabled_config, sample_cases
    ):
        """evaluate_case returns DiagnosisResult."""
        engine = BeliefRevisionEngine(mock_openai_client, premortem_enabled_config)
        case = sample_cases["simple"]

        result = engine.evaluate_case(
            case_text_20pct=case["case_20pct"],
            case_text_full=case["case_full"],
            task_type="free_text"
        )

        assert isinstance(result, DiagnosisResult)

    def test_result_contains_final_diagnosis(
        self, mock_openai_client, premortem_enabled_config, sample_cases
    ):
        """Result contains final diagnosis."""
        engine = BeliefRevisionEngine(mock_openai_client, premortem_enabled_config)
        case = sample_cases["simple"]

        result = engine.evaluate_case(
            case_text_20pct=case["case_20pct"],
            case_text_full=case["case_full"],
            task_type="free_text"
        )

        assert result.final_diagnosis is not None
        assert isinstance(result.final_diagnosis, str)

    def test_result_contains_initial_hypothesis(
        self, mock_openai_client, premortem_enabled_config, sample_cases
    ):
        """Result contains initial hypothesis."""
        engine = BeliefRevisionEngine(mock_openai_client, premortem_enabled_config)
        case = sample_cases["simple"]

        result = engine.evaluate_case(
            case_text_20pct=case["case_20pct"],
            case_text_full=case["case_full"],
            task_type="free_text"
        )

        assert result.initial_hypothesis is not None
        assert isinstance(result.initial_hypothesis, str)

    def test_result_contains_quadrant(
        self, mock_openai_client, premortem_enabled_config, sample_cases
    ):
        """Result contains quadrant classification."""
        engine = BeliefRevisionEngine(mock_openai_client, premortem_enabled_config)
        case = sample_cases["simple"]

        result = engine.evaluate_case(
            case_text_20pct=case["case_20pct"],
            case_text_full=case["case_full"],
            task_type="free_text"
        )

        assert isinstance(result.quadrant, RiskQuadrant)
        assert result.quadrant_result is not None

    def test_result_contains_latency(
        self, mock_openai_client, premortem_enabled_config, sample_cases
    ):
        """Result tracks latency metrics."""
        engine = BeliefRevisionEngine(mock_openai_client, premortem_enabled_config)
        case = sample_cases["simple"]

        result = engine.evaluate_case(
            case_text_20pct=case["case_20pct"],
            case_text_full=case["case_full"],
            task_type="free_text"
        )

        assert 'pass1' in result.latency_ms
        assert 'pass2' in result.latency_ms
        assert 'quadrant' in result.latency_ms
        assert all(v >= 0 for v in result.latency_ms.values())


class TestPreMortemTriggering:
    """Tests for Pre-Mortem triggering behavior."""

    def test_premortem_not_applied_when_disabled(
        self, mock_openai_client, sample_cases
    ):
        """Pre-Mortem not applied when disabled."""
        config = PreMortemConfig(enable_premortem=False)
        engine = BeliefRevisionEngine(mock_openai_client, config)
        case = sample_cases["complex_high_stakes"]

        result = engine.evaluate_case(
            case_text_20pct=case["case_20pct"],
            case_text_full=case["case_full"],
            task_type="free_text"
        )

        assert result.premortem_applied is False
        assert result.premortem_result is None

    def test_premortem_result_when_applied(self, mock_openai_client_with_responses, sample_cases):
        """Pre-Mortem result populated when applied."""
        client, set_responses = mock_openai_client_with_responses

        # Setup responses for: initial hypothesis, pre-mortem, final diagnosis
        set_responses([
            "Initial diagnosis",
            """ALTERNATIVE_DIAGNOSIS: Alternative
EVIDENCE: Some evidence
EVIDENCE_STRENGTH: 0.6
MISSED_RED_FLAGS: None
RECOMMENDATION: MAINTAIN""",
            "Final diagnosis"
        ])

        # Use low thresholds to trigger Pre-Mortem
        config = PreMortemConfig(
            enable_premortem=True,
            complexity_threshold=0.1,
            stakes_threshold=0.1
        )
        engine = BeliefRevisionEngine(client, config)
        case = sample_cases["complex_high_stakes"]

        result = engine.evaluate_case(
            case_text_20pct=case["case_20pct"],
            case_text_full=case["case_full"],
            task_type="free_text"
        )

        # May or may not trigger depending on classification
        if result.premortem_applied:
            assert result.premortem_result is not None
            assert isinstance(result.premortem_result, PreMortemResponse)


class TestMCQEvaluation:
    """Tests for MCQ task type."""

    def test_mcq_with_options(
        self, mock_openai_client_with_responses, premortem_enabled_config, sample_cases
    ):
        """MCQ evaluation works with options."""
        client, set_responses = mock_openai_client_with_responses
        set_responses(["1", "1"])  # MCQ responses for pass 1 and pass 2

        engine = BeliefRevisionEngine(client, premortem_enabled_config)
        case = sample_cases["simple"]
        options = [case["true_diagnosis"]] + case["distractors"]

        result = engine.evaluate_case(
            case_text_20pct=case["case_20pct"],
            case_text_full=case["case_full"],
            task_type="mcq",
            options=options
        )

        assert result.final_diagnosis is not None


class TestBeliefRevision:
    """Tests for belief revision detection."""

    def test_detects_no_revision(self, mock_openai_client_with_responses, sample_cases):
        """Detects when no revision occurs."""
        client, set_responses = mock_openai_client_with_responses
        set_responses(["Same diagnosis", "Same diagnosis"])

        config = PreMortemConfig(enable_premortem=False)
        engine = BeliefRevisionEngine(client, config)
        case = sample_cases["simple"]

        result = engine.evaluate_case(
            case_text_20pct=case["case_20pct"],
            case_text_full=case["case_full"],
            task_type="free_text"
        )

        assert result.belief_revision_occurred is False

    def test_detects_revision(self, mock_openai_client_with_responses, sample_cases):
        """Detects when revision occurs."""
        client, set_responses = mock_openai_client_with_responses
        set_responses(["Initial diagnosis", "Different diagnosis"])

        config = PreMortemConfig(enable_premortem=False)
        engine = BeliefRevisionEngine(client, config)
        case = sample_cases["simple"]

        result = engine.evaluate_case(
            case_text_20pct=case["case_20pct"],
            case_text_full=case["case_full"],
            task_type="free_text"
        )

        assert result.belief_revision_occurred is True

    def test_mcq_revision_detection(self, mock_openai_client_with_responses, sample_cases):
        """Detects revision in MCQ based on number change."""
        client, set_responses = mock_openai_client_with_responses
        set_responses(["1", "2"])  # Different numbers = revision

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

        assert result.belief_revision_occurred is True


class TestConfidenceScores:
    """Tests for confidence scoring."""

    def test_initial_confidence_default(
        self, mock_openai_client, premortem_enabled_config, sample_cases
    ):
        """Initial confidence has default value."""
        engine = BeliefRevisionEngine(mock_openai_client, premortem_enabled_config)
        case = sample_cases["simple"]

        result = engine.evaluate_case(
            case_text_20pct=case["case_20pct"],
            case_text_full=case["case_full"],
            task_type="free_text"
        )

        assert 0 <= result.initial_confidence <= 1

    def test_final_confidence_default(
        self, mock_openai_client, premortem_enabled_config, sample_cases
    ):
        """Final confidence has default value."""
        engine = BeliefRevisionEngine(mock_openai_client, premortem_enabled_config)
        case = sample_cases["simple"]

        result = engine.evaluate_case(
            case_text_20pct=case["case_20pct"],
            case_text_full=case["case_full"],
            task_type="free_text"
        )

        assert 0 <= result.final_confidence <= 1

    def test_revision_magnitude_calculated(
        self, mock_openai_client, premortem_enabled_config, sample_cases
    ):
        """Revision magnitude is calculated."""
        engine = BeliefRevisionEngine(mock_openai_client, premortem_enabled_config)
        case = sample_cases["simple"]

        result = engine.evaluate_case(
            case_text_20pct=case["case_20pct"],
            case_text_full=case["case_full"],
            task_type="free_text"
        )

        assert result.revision_magnitude >= 0


class TestConvenienceFunction:
    """Tests for evaluate_case_with_premortem convenience function."""

    def test_convenience_function_works(
        self, mock_openai_client, sample_cases
    ):
        """Convenience function returns result."""
        case = sample_cases["simple"]

        result = evaluate_case_with_premortem(
            llm_client=mock_openai_client,
            case_text_20pct=case["case_20pct"],
            case_text_full=case["case_full"],
            task_type="free_text"
        )

        assert isinstance(result, DiagnosisResult)

    def test_convenience_function_with_config(
        self, mock_openai_client, premortem_enabled_config, sample_cases
    ):
        """Convenience function accepts config."""
        case = sample_cases["simple"]

        result = evaluate_case_with_premortem(
            llm_client=mock_openai_client,
            case_text_20pct=case["case_20pct"],
            case_text_full=case["case_full"],
            task_type="free_text",
            config=premortem_enabled_config
        )

        assert isinstance(result, DiagnosisResult)


class TestDiagnosisResultDataclass:
    """Tests for DiagnosisResult dataclass structure."""

    def test_result_has_all_fields(
        self, mock_openai_client, premortem_enabled_config, sample_cases
    ):
        """DiagnosisResult has all expected fields."""
        engine = BeliefRevisionEngine(mock_openai_client, premortem_enabled_config)
        case = sample_cases["simple"]

        result = engine.evaluate_case(
            case_text_20pct=case["case_20pct"],
            case_text_full=case["case_full"],
            task_type="free_text"
        )

        # Check all required fields exist
        assert hasattr(result, 'final_diagnosis')
        assert hasattr(result, 'final_confidence')
        assert hasattr(result, 'initial_hypothesis')
        assert hasattr(result, 'initial_confidence')
        assert hasattr(result, 'quadrant')
        assert hasattr(result, 'quadrant_result')
        assert hasattr(result, 'premortem_applied')
        assert hasattr(result, 'premortem_result')
        assert hasattr(result, 'belief_revision_occurred')
        assert hasattr(result, 'revision_magnitude')
        assert hasattr(result, 'tokens_used')
        assert hasattr(result, 'latency_ms')


class TestErrorHandling:
    """Tests for error handling."""

    def test_handles_api_error_with_retry(self, sample_cases):
        """Handles API errors with retry logic."""
        client = MagicMock()
        call_count = [0]

        def mock_api(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] < 2:
                raise Exception("API Error")
            response = MagicMock()
            response.choices = [MagicMock()]
            response.choices[0].message.content = "diagnosis"
            return response

        client.chat.completions.create = mock_api

        config = PreMortemConfig(enable_premortem=False)
        engine = BeliefRevisionEngine(client, config)
        case = sample_cases["simple"]

        # Should succeed after retry
        with patch('time.sleep'):  # Skip actual sleep
            result = engine.evaluate_case(
                case_text_20pct=case["case_20pct"],
                case_text_full=case["case_full"],
                task_type="free_text"
            )

        assert result.final_diagnosis is not None


class TestVerboseMode:
    """Tests for verbose mode output."""

    def test_verbose_mode_runs(
        self, mock_openai_client, sample_cases, capsys
    ):
        """Verbose mode produces output."""
        config = PreMortemConfig(enable_premortem=False, verbose=True)
        engine = BeliefRevisionEngine(mock_openai_client, config)
        case = sample_cases["simple"]

        result = engine.evaluate_case(
            case_text_20pct=case["case_20pct"],
            case_text_full=case["case_full"],
            task_type="free_text"
        )

        captured = capsys.readouterr()
        assert "[Pass 1]" in captured.out
        assert "[Quadrant]" in captured.out
        assert "[Pass 2]" in captured.out
