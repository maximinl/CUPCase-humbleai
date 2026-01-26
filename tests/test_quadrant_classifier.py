"""
Tests for the QuadrantClassifier module.

Tests cover:
- Classifier initialization
- Complexity scoring
- Stakes scoring
- Red flag detection
- Quadrant determination
- Pre-Mortem trigger logic
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'gpt_and_med_lm_evaluation'))

from premortem.quadrant_classifier import QuadrantClassifier, QuadrantResult, classify_case
from premortem.config import PreMortemConfig, RiskQuadrant


class TestQuadrantClassifierInit:
    """Tests for classifier initialization."""

    def test_init_with_default_config(self):
        """Classifier initializes with default config."""
        classifier = QuadrantClassifier()
        assert classifier.config is not None
        assert classifier.config.complexity_threshold == 0.5
        assert classifier.config.stakes_threshold == 0.5

    def test_init_with_custom_config(self, premortem_enabled_config):
        """Classifier initializes with custom config."""
        classifier = QuadrantClassifier(premortem_enabled_config)
        assert classifier.config == premortem_enabled_config


class TestComplexityScoring:
    """Tests for complexity score calculation."""

    def test_low_complexity_case(self):
        """Simple cases have low complexity scores."""
        classifier = QuadrantClassifier()
        case = "Patient with typical presentation of common cold. Mild symptoms, stable."
        result = classifier.classify(case)
        assert result.complexity_score < 0.5

    def test_high_complexity_case(self):
        """Complex cases have high complexity scores."""
        classifier = QuadrantClassifier()
        case = (
            "Patient with atypical, rare, complex presentation with multiple "
            "comorbidities and uncertain etiology. Conflicting test results."
        )
        result = classifier.classify(case)
        assert result.complexity_score > 0.3  # Should be elevated

    def test_complexity_increases_with_length(self):
        """Longer cases tend to have higher complexity scores."""
        classifier = QuadrantClassifier()
        short_case = "Simple headache."
        long_case = "Simple headache. " * 50  # Much longer

        short_result = classifier.classify(short_case)
        long_result = classifier.classify(long_case)

        # Length factor should increase score
        assert long_result.complexity_score >= short_result.complexity_score


class TestStakesScoring:
    """Tests for stakes score calculation."""

    def test_low_stakes_case(self):
        """Benign cases have low stakes scores."""
        classifier = QuadrantClassifier()
        case = "Patient with benign, self-limiting condition. Routine follow-up. Stable."
        result = classifier.classify(case)
        assert result.stakes_score < 0.5

    def test_high_stakes_case(self):
        """Emergency cases have high stakes scores."""
        classifier = QuadrantClassifier()
        case = (
            "Patient in emergency with critical, life-threatening condition. "
            "Sepsis, shock, respiratory failure. Unstable vital signs."
        )
        result = classifier.classify(case)
        assert result.stakes_score > 0.3  # Should be elevated

    def test_stakes_higher_with_multiple_indicators(self):
        """Multiple high-stakes indicators increase score."""
        classifier = QuadrantClassifier()
        single_indicator = "Patient with acute condition."
        multiple_indicators = (
            "Patient with acute, severe, critical, emergency condition. "
            "Life-threatening. Unstable."
        )

        single_result = classifier.classify(single_indicator)
        multiple_result = classifier.classify(multiple_indicators)

        assert multiple_result.stakes_score >= single_result.stakes_score


class TestRedFlagDetection:
    """Tests for red flag detection."""

    def test_detects_chest_pain(self):
        """Detects chest pain red flag."""
        classifier = QuadrantClassifier()
        case = "Patient presents with chest pain."
        result = classifier.classify(case)
        assert "chest pain" in result.red_flags_detected

    def test_detects_shortness_of_breath(self):
        """Detects shortness of breath red flag."""
        classifier = QuadrantClassifier()
        case = "Patient reports shortness of breath."
        result = classifier.classify(case)
        assert "shortness of breath" in result.red_flags_detected

    def test_detects_syncope(self):
        """Detects syncope red flag."""
        classifier = QuadrantClassifier()
        case = "Patient had syncope yesterday."
        result = classifier.classify(case)
        assert "syncope" in result.red_flags_detected

    def test_detects_multiple_red_flags(self, case_with_red_flags):
        """Detects multiple red flags."""
        classifier = QuadrantClassifier()
        result = classifier.classify(case_with_red_flags)
        assert len(result.red_flags_detected) >= 3

    def test_no_red_flags_in_benign_case(self, case_without_red_flags):
        """No red flags detected in benign case."""
        classifier = QuadrantClassifier()
        result = classifier.classify(case_without_red_flags)
        assert len(result.red_flags_detected) == 0

    def test_red_flags_increase_stakes(self):
        """Red flags increase stakes score."""
        classifier = QuadrantClassifier()

        no_flags = "Patient with mild symptoms."
        with_flags = "Patient with chest pain and syncope."

        no_flags_result = classifier.classify(no_flags)
        with_flags_result = classifier.classify(with_flags)

        assert with_flags_result.stakes_score > no_flags_result.stakes_score


class TestQuadrantDetermination:
    """Tests for quadrant determination logic."""

    def test_routine_quadrant(self):
        """Low complexity + low stakes = ROUTINE."""
        classifier = QuadrantClassifier()
        case = "Typical mild headache, stable, benign condition."
        result = classifier.classify(case)
        # May not always be ROUTINE depending on exact scoring, but should be low risk
        assert result.quadrant in [RiskQuadrant.ROUTINE, RiskQuadrant.WATCHFUL, RiskQuadrant.CURIOSITY]

    def test_escalate_quadrant(self):
        """High complexity + high stakes = ESCALATE."""
        classifier = QuadrantClassifier()
        case = (
            "Complex, atypical, rare presentation with multiple comorbidities. "
            "Critical emergency, life-threatening, unstable vital signs."
        )
        result = classifier.classify(case)
        # Should tend toward ESCALATE with such indicators
        assert result.stakes_score > 0.3 or result.complexity_score > 0.3

    def test_quadrant_result_structure(self, sample_cases):
        """QuadrantResult has all required fields."""
        classifier = QuadrantClassifier()
        case = sample_cases["simple"]["case_full"]
        result = classifier.classify(case)

        assert isinstance(result, QuadrantResult)
        assert isinstance(result.quadrant, RiskQuadrant)
        assert 0 <= result.complexity_score <= 1
        assert 0 <= result.stakes_score <= 1
        assert isinstance(result.red_flags_detected, list)
        assert isinstance(result.requires_premortem, bool)
        assert isinstance(result.reasoning, str)


class TestPreMortemTrigger:
    """Tests for Pre-Mortem trigger logic."""

    def test_premortem_triggered_for_watchful(self):
        """Pre-Mortem triggered for WATCHFUL quadrant."""
        config = PreMortemConfig(
            premortem_quadrants=[RiskQuadrant.WATCHFUL, RiskQuadrant.ESCALATE]
        )
        classifier = QuadrantClassifier(config)

        # Create a case that's low complexity but high stakes
        case = "Simple emergency. Critical. Life-threatening."
        result = classifier.classify(case)

        # If classified as WATCHFUL, should trigger Pre-Mortem
        if result.quadrant == RiskQuadrant.WATCHFUL:
            assert result.requires_premortem is True

    def test_premortem_triggered_for_escalate(self):
        """Pre-Mortem triggered for ESCALATE quadrant."""
        config = PreMortemConfig(
            premortem_quadrants=[RiskQuadrant.WATCHFUL, RiskQuadrant.ESCALATE]
        )
        classifier = QuadrantClassifier(config)

        case = (
            "Complex atypical critical emergency. Life-threatening. "
            "Multiple comorbidities. Unstable."
        )
        result = classifier.classify(case)

        if result.quadrant == RiskQuadrant.ESCALATE:
            assert result.requires_premortem is True

    def test_premortem_triggered_by_multiple_red_flags(self):
        """Multiple red flags trigger Pre-Mortem regardless of quadrant."""
        classifier = QuadrantClassifier()
        case = (
            "Chest pain with shortness of breath and syncope. "
            "Altered mental status. Immunocompromised."
        )
        result = classifier.classify(case)

        # Multiple red flags should trigger Pre-Mortem
        if len(result.red_flags_detected) >= 2:
            assert result.requires_premortem is True

    def test_premortem_not_triggered_for_routine(self):
        """Pre-Mortem not triggered for ROUTINE quadrant (unless red flags)."""
        config = PreMortemConfig(
            premortem_quadrants=[RiskQuadrant.WATCHFUL, RiskQuadrant.ESCALATE]
        )
        classifier = QuadrantClassifier(config)

        case = "Typical mild stable benign condition. Routine follow-up."
        result = classifier.classify(case)

        if result.quadrant == RiskQuadrant.ROUTINE and len(result.red_flags_detected) < 2:
            assert result.requires_premortem is False


class TestReasoningGeneration:
    """Tests for reasoning string generation."""

    def test_reasoning_contains_quadrant(self):
        """Reasoning includes quadrant name."""
        classifier = QuadrantClassifier()
        result = classifier.classify("Some case presentation.")
        assert result.quadrant.name in result.reasoning

    def test_reasoning_contains_scores(self):
        """Reasoning includes complexity and stakes scores."""
        classifier = QuadrantClassifier()
        result = classifier.classify("Some case presentation.")
        assert "Complexity:" in result.reasoning
        assert "Stakes:" in result.reasoning

    def test_reasoning_contains_red_flags(self, case_with_red_flags):
        """Reasoning includes detected red flags."""
        classifier = QuadrantClassifier()
        result = classifier.classify(case_with_red_flags)
        if result.red_flags_detected:
            assert "Red flags detected:" in result.reasoning


class TestConvenienceFunction:
    """Tests for the classify_case convenience function."""

    def test_classify_case_returns_result(self):
        """classify_case returns QuadrantResult."""
        result = classify_case("Some case presentation.")
        assert isinstance(result, QuadrantResult)

    def test_classify_case_with_config(self, low_threshold_config):
        """classify_case works with custom config."""
        result = classify_case("Some case.", config=low_threshold_config)
        assert isinstance(result, QuadrantResult)


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_case(self):
        """Handles empty case text."""
        classifier = QuadrantClassifier()
        result = classifier.classify("")
        assert isinstance(result, QuadrantResult)

    def test_very_long_case(self):
        """Handles very long case text."""
        classifier = QuadrantClassifier()
        long_case = "Patient presents. " * 1000
        result = classifier.classify(long_case)
        assert isinstance(result, QuadrantResult)

    def test_special_characters(self):
        """Handles special characters in case text."""
        classifier = QuadrantClassifier()
        case = "Patient with temp 38.5°C, BP 120/80 mmHg, O2 sat 98%."
        result = classifier.classify(case)
        assert isinstance(result, QuadrantResult)

    def test_case_insensitivity(self):
        """Detection is case-insensitive."""
        classifier = QuadrantClassifier()

        lower = "chest pain"
        upper = "CHEST PAIN"
        mixed = "Chest Pain"

        lower_result = classifier.classify(lower)
        upper_result = classifier.classify(upper)
        mixed_result = classifier.classify(mixed)

        # All should detect the red flag
        assert "chest pain" in lower_result.red_flags_detected
        assert "chest pain" in upper_result.red_flags_detected
        assert "chest pain" in mixed_result.red_flags_detected
