"""
Tests for the PreMortemPrompts module.

Tests cover:
- Prompt building
- Response parsing
- Edge cases and error handling
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'gpt_and_med_lm_evaluation'))

from premortem.premortem_prompts import PreMortemPrompts, PreMortemResponse


class TestPreMortemPromptBuilding:
    """Tests for Pre-Mortem prompt construction."""

    def test_premortem_prompt_contains_case(self):
        """Prompt contains the case text."""
        case = "Patient with symptoms."
        prompt = PreMortemPrompts.build_premortem_prompt(
            case_text=case,
            hypothesis="Some diagnosis"
        )
        assert case in prompt

    def test_premortem_prompt_contains_hypothesis(self):
        """Prompt contains the hypothesis."""
        hypothesis = "Acute myocardial infarction"
        prompt = PreMortemPrompts.build_premortem_prompt(
            case_text="Some case",
            hypothesis=hypothesis
        )
        assert hypothesis in prompt

    def test_premortem_prompt_contains_token_percentage(self):
        """Prompt contains the token percentage."""
        prompt = PreMortemPrompts.build_premortem_prompt(
            case_text="Some case",
            hypothesis="Some diagnosis",
            token_percentage=20.0
        )
        assert "20%" in prompt

    def test_premortem_prompt_contains_key_instructions(self):
        """Prompt contains key Pre-Mortem instructions."""
        prompt = PreMortemPrompts.build_premortem_prompt(
            case_text="Some case",
            hypothesis="Some diagnosis"
        )
        assert "WRONG" in prompt
        assert "dangerous" in prompt.lower()
        assert "ALTERNATIVE_DIAGNOSIS" in prompt
        assert "EVIDENCE_STRENGTH" in prompt
        assert "RECOMMENDATION" in prompt


class TestBeliefRevisionPromptBuilding:
    """Tests for belief revision prompt construction."""

    def test_belief_revision_prompt_mcq(self):
        """MCQ belief revision prompt includes options."""
        options = ["Option A", "Option B", "Option C", "Option D"]
        prompt = PreMortemPrompts.build_belief_revision_prompt(
            full_case="Full case text",
            initial_hypothesis="Initial diagnosis",
            premortem_alternative="Alternative diagnosis",
            evidence_strength=0.7,
            premortem_recommendation="REVISE",
            task_type="mcq",
            options=options
        )

        for opt in options:
            assert opt in prompt
        assert "1." in prompt or "1)" in prompt

    def test_belief_revision_prompt_free_text(self):
        """Free-text belief revision prompt has correct format."""
        prompt = PreMortemPrompts.build_belief_revision_prompt(
            full_case="Full case text",
            initial_hypothesis="Initial diagnosis",
            premortem_alternative="Alternative diagnosis",
            evidence_strength=0.5,
            premortem_recommendation="MAINTAIN",
            task_type="free_text"
        )

        assert "concise sentence" in prompt.lower()
        assert "Full case text" in prompt

    def test_belief_revision_contains_premortem_info(self):
        """Prompt contains Pre-Mortem analysis information."""
        prompt = PreMortemPrompts.build_belief_revision_prompt(
            full_case="Case",
            initial_hypothesis="Hypothesis",
            premortem_alternative="Alternative",
            evidence_strength=0.8,
            premortem_recommendation="ESCALATE",
            task_type="free_text"
        )

        assert "Hypothesis" in prompt
        assert "Alternative" in prompt
        assert "0.8" in prompt
        assert "ESCALATE" in prompt


class TestInitialHypothesisPrompt:
    """Tests for initial hypothesis prompt building."""

    def test_initial_hypothesis_mcq(self):
        """MCQ initial hypothesis prompt includes options."""
        options = ["A", "B", "C", "D"]
        prompt = PreMortemPrompts.build_initial_hypothesis_prompt(
            case_text="Case text",
            task_type="mcq",
            options=options
        )

        for opt in options:
            assert opt in prompt
        assert "Initial assessment" in prompt

    def test_initial_hypothesis_free_text(self):
        """Free-text initial hypothesis prompt has correct format."""
        prompt = PreMortemPrompts.build_initial_hypothesis_prompt(
            case_text="Case text",
            task_type="free_text"
        )

        assert "concise sentence" in prompt.lower()
        assert "Initial assessment" in prompt


class TestFinalDiagnosisPrompt:
    """Tests for final diagnosis prompt building."""

    def test_final_diagnosis_mcq(self):
        """MCQ final diagnosis prompt includes options."""
        options = ["A", "B", "C", "D"]
        prompt = PreMortemPrompts.build_final_diagnosis_prompt(
            case_text="Case text",
            task_type="mcq",
            options=options
        )

        for opt in options:
            assert opt in prompt
        assert "Full case review" in prompt

    def test_final_diagnosis_free_text(self):
        """Free-text final diagnosis prompt has correct format."""
        prompt = PreMortemPrompts.build_final_diagnosis_prompt(
            case_text="Case text",
            task_type="free_text"
        )

        assert "concise sentence" in prompt.lower()
        assert "Full case review" in prompt


class TestPreMortemResponseParsing:
    """Tests for parsing Pre-Mortem responses."""

    def test_parse_complete_response(self, premortem_response_text):
        """Parses a complete well-formed response."""
        result = PreMortemPrompts.parse_premortem_response(premortem_response_text)

        assert isinstance(result, PreMortemResponse)
        assert result.alternative_diagnosis == "Pulmonary embolism"
        assert "dyspnea" in result.evidence_for_alternative.lower()
        assert result.evidence_strength == 0.75
        assert "tachycardia" in result.missed_red_flags
        assert result.recommendation == "REVISE"
        assert result.raw_response == premortem_response_text

    def test_parse_incomplete_response(self, premortem_response_incomplete):
        """Handles incomplete responses gracefully."""
        result = PreMortemPrompts.parse_premortem_response(premortem_response_incomplete)

        assert isinstance(result, PreMortemResponse)
        assert result.alternative_diagnosis == "Some diagnosis"
        # Invalid evidence strength should default to 0.5
        assert result.evidence_strength == 0.5
        assert result.recommendation == "ESCALATE"

    def test_parse_empty_response(self):
        """Handles empty response."""
        result = PreMortemPrompts.parse_premortem_response("")

        assert isinstance(result, PreMortemResponse)
        assert result.alternative_diagnosis == ""
        assert result.evidence_strength == 0.5
        assert result.recommendation == "MAINTAIN"

    def test_parse_response_with_extra_text(self):
        """Handles responses with extra explanatory text."""
        response = """Let me analyze this case...

ALTERNATIVE_DIAGNOSIS: Aortic dissection
EVIDENCE: Tearing chest pain, hypertension
EVIDENCE_STRENGTH: 0.8
MISSED_RED_FLAGS: blood pressure differential
RECOMMENDATION: ESCALATE

This is my analysis because..."""

        result = PreMortemPrompts.parse_premortem_response(response)

        assert result.alternative_diagnosis == "Aortic dissection"
        assert result.evidence_strength == 0.8
        assert result.recommendation == "ESCALATE"

    def test_parse_evidence_strength_clamping(self):
        """Evidence strength is clamped to [0, 1]."""
        response_high = "EVIDENCE_STRENGTH: 1.5"
        response_low = "EVIDENCE_STRENGTH: -0.2"

        result_high = PreMortemPrompts.parse_premortem_response(response_high)
        result_low = PreMortemPrompts.parse_premortem_response(response_low)

        assert 0 <= result_high.evidence_strength <= 1
        assert 0 <= result_low.evidence_strength <= 1

    def test_parse_recommendation_extraction(self):
        """Correctly extracts recommendation keywords."""
        for rec in ["MAINTAIN", "REVISE", "ESCALATE"]:
            response = f"RECOMMENDATION: {rec}"
            result = PreMortemPrompts.parse_premortem_response(response)
            assert result.recommendation == rec

    def test_parse_recommendation_with_explanation(self):
        """Extracts recommendation even with explanation."""
        response = "RECOMMENDATION: ESCALATE (due to high risk)"
        result = PreMortemPrompts.parse_premortem_response(response)
        assert result.recommendation == "ESCALATE"

    def test_parse_red_flags_none(self):
        """Handles 'None' red flags."""
        response = "MISSED_RED_FLAGS: None"
        result = PreMortemPrompts.parse_premortem_response(response)
        assert result.missed_red_flags == []

    def test_parse_red_flags_na(self):
        """Handles 'N/A' red flags."""
        response = "MISSED_RED_FLAGS: N/A"
        result = PreMortemPrompts.parse_premortem_response(response)
        assert result.missed_red_flags == []

    def test_parse_multiple_red_flags(self):
        """Parses comma-separated red flags."""
        response = "MISSED_RED_FLAGS: fever, tachycardia, hypotension"
        result = PreMortemPrompts.parse_premortem_response(response)
        assert len(result.missed_red_flags) == 3
        assert "fever" in result.missed_red_flags
        assert "tachycardia" in result.missed_red_flags
        assert "hypotension" in result.missed_red_flags


class TestPreMortemResponseDataclass:
    """Tests for PreMortemResponse dataclass."""

    def test_response_fields(self):
        """Response has all expected fields."""
        response = PreMortemResponse(
            alternative_diagnosis="Test diagnosis",
            evidence_for_alternative="Test evidence",
            evidence_strength=0.6,
            missed_red_flags=["flag1", "flag2"],
            recommendation="MAINTAIN",
            raw_response="raw"
        )

        assert response.alternative_diagnosis == "Test diagnosis"
        assert response.evidence_for_alternative == "Test evidence"
        assert response.evidence_strength == 0.6
        assert response.missed_red_flags == ["flag1", "flag2"]
        assert response.recommendation == "MAINTAIN"
        assert response.raw_response == "raw"


class TestEdgeCases:
    """Tests for edge cases."""

    def test_prompt_with_special_characters(self):
        """Handles special characters in prompts."""
        case = "Temp: 38.5°C, BP: 120/80 mmHg, SpO₂: 98%"
        prompt = PreMortemPrompts.build_premortem_prompt(
            case_text=case,
            hypothesis="Sepsis"
        )
        assert case in prompt

    def test_prompt_with_long_text(self):
        """Handles very long text."""
        long_case = "Patient details. " * 500
        prompt = PreMortemPrompts.build_premortem_prompt(
            case_text=long_case,
            hypothesis="Diagnosis"
        )
        assert len(prompt) > len(long_case)

    def test_prompt_with_unicode(self):
        """Handles unicode characters."""
        case = "患者表现为头痛 (Patient presents with headache)"
        prompt = PreMortemPrompts.build_premortem_prompt(
            case_text=case,
            hypothesis="Headache"
        )
        assert case in prompt

    def test_parse_malformed_response(self):
        """Handles completely malformed responses."""
        malformed = "This is just some random text without proper formatting."
        result = PreMortemPrompts.parse_premortem_response(malformed)

        # Should return default values
        assert isinstance(result, PreMortemResponse)
        assert result.recommendation == "MAINTAIN"  # Default
