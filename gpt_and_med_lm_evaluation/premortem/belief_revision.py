"""
Belief Revision Engine.

Orchestrates the complete Pre-Mortem evaluation pipeline:
1. Pass 1: Generate initial hypothesis with partial case
2. Quadrant Classification: Assess risk level
3. Pre-Mortem Analysis (if triggered): Challenge the hypothesis
4. Pass 2: Generate final diagnosis with full case and belief revision
"""

import time
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field

from .config import PreMortemConfig, RiskQuadrant
from .quadrant_classifier import QuadrantClassifier, QuadrantResult
from .premortem_prompts import PreMortemPrompts, PreMortemResponse


@dataclass
class DiagnosisResult:
    """
    Complete result from the Pre-Mortem evaluation pipeline.

    Contains all information from each phase of the evaluation,
    including metrics for analysis.

    Attributes:
        final_diagnosis: The final diagnosis output
        final_confidence: Confidence in final diagnosis (0-1)
        initial_hypothesis: Initial hypothesis from Pass 1
        initial_confidence: Confidence in initial hypothesis (0-1)
        quadrant: Assigned risk quadrant
        quadrant_result: Full QuadrantResult object
        premortem_applied: Whether Pre-Mortem was triggered
        premortem_result: PreMortemResponse if Pre-Mortem was applied
        belief_revision_occurred: Whether the diagnosis changed
        revision_magnitude: Magnitude of confidence change
        tokens_used: Token counts per phase
        latency_ms: Latency measurements per phase
    """
    # Final result
    final_diagnosis: str
    final_confidence: float

    # Pass 1 information
    initial_hypothesis: str
    initial_confidence: float

    # Quadrant information
    quadrant: RiskQuadrant
    quadrant_result: QuadrantResult

    # Pre-Mortem information (if applied)
    premortem_applied: bool
    premortem_result: Optional[PreMortemResponse]

    # Revision metrics
    belief_revision_occurred: bool
    revision_magnitude: float

    # Performance metrics
    tokens_used: Dict[str, int] = field(default_factory=dict)
    latency_ms: Dict[str, float] = field(default_factory=dict)


class BeliefRevisionEngine:
    """
    Engine that orchestrates the Pre-Mortem evaluation pipeline.

    The pipeline consists of:
    1. Pass 1: Generate initial hypothesis using partial case (20% tokens)
    2. Quadrant Classification: Determine risk level
    3. Pre-Mortem (conditional): Challenge the hypothesis if in high-risk quadrant
    4. Pass 2: Generate final diagnosis using full case

    Usage:
        from openai import OpenAI
        client = OpenAI(api_key="...")
        config = PreMortemConfig(enable_premortem=True)
        engine = BeliefRevisionEngine(client, config)
        result = engine.evaluate_case(
            case_text_20pct="partial case...",
            case_text_full="full case...",
            task_type="mcq",
            options=["A", "B", "C", "D"]
        )
    """

    def __init__(self, llm_client: Any, config: Optional[PreMortemConfig] = None):
        """
        Initialize the engine.

        Args:
            llm_client: OpenAI client instance
            config: PreMortemConfig instance (uses defaults if not provided)
        """
        self.client = llm_client
        self.config = config or PreMortemConfig()
        self.classifier = QuadrantClassifier(self.config)

    def evaluate_case(
        self,
        case_text_20pct: str,
        case_text_full: str,
        task_type: str = "free_text",
        options: Optional[List[str]] = None,
        true_diagnosis: Optional[str] = None
    ) -> DiagnosisResult:
        """
        Evaluate a clinical case through the complete pipeline.

        Args:
            case_text_20pct: Case text truncated to 20% of tokens
            case_text_full: Complete case text
            task_type: "mcq" for multiple choice, "free_text" for open-ended
            options: List of MCQ options (required if task_type is "mcq")
            true_diagnosis: Ground truth diagnosis (optional, for metrics)

        Returns:
            DiagnosisResult containing all evaluation information
        """
        tokens_used: Dict[str, int] = {}
        latency: Dict[str, float] = {}

        # ===== PASS 1: Initial Hypothesis with 20% tokens =====
        t0 = time.time()
        initial_hypothesis, initial_confidence = self._generate_initial_hypothesis(
            case_text_20pct, task_type, options
        )
        latency['pass1'] = (time.time() - t0) * 1000

        if self.config.verbose:
            print(f"[Pass 1] Initial hypothesis: {initial_hypothesis}")

        # ===== QUADRANT CLASSIFICATION =====
        t0 = time.time()
        quadrant_result = self.classifier.classify(case_text_20pct, initial_hypothesis)
        latency['quadrant'] = (time.time() - t0) * 1000

        if self.config.verbose:
            print(f"[Quadrant] {quadrant_result.reasoning}")

        # ===== PRE-MORTEM ANALYSIS (if triggered) =====
        premortem_result: Optional[PreMortemResponse] = None

        if self.config.enable_premortem and quadrant_result.requires_premortem:
            t0 = time.time()
            premortem_result = self._run_premortem(
                case_text_20pct, initial_hypothesis
            )
            latency['premortem'] = (time.time() - t0) * 1000

            if self.config.verbose:
                print(f"[Pre-Mortem] Alternative: {premortem_result.alternative_diagnosis}")
                print(f"[Pre-Mortem] Recommendation: {premortem_result.recommendation}")

        # ===== PASS 2: Final Diagnosis with Full Case =====
        t0 = time.time()
        final_diagnosis, final_confidence = self._generate_final_diagnosis(
            case_text_full,
            initial_hypothesis,
            initial_confidence,
            premortem_result,
            task_type,
            options
        )
        latency['pass2'] = (time.time() - t0) * 1000

        if self.config.verbose:
            print(f"[Pass 2] Final diagnosis: {final_diagnosis}")

        # ===== CALCULATE REVISION METRICS =====
        belief_revision = self._diagnoses_differ(initial_hypothesis, final_diagnosis)
        revision_magnitude = abs(final_confidence - initial_confidence)

        return DiagnosisResult(
            final_diagnosis=final_diagnosis,
            final_confidence=final_confidence,
            initial_hypothesis=initial_hypothesis,
            initial_confidence=initial_confidence,
            quadrant=quadrant_result.quadrant,
            quadrant_result=quadrant_result,
            premortem_applied=premortem_result is not None,
            premortem_result=premortem_result,
            belief_revision_occurred=belief_revision,
            revision_magnitude=revision_magnitude,
            tokens_used=tokens_used,
            latency_ms=latency
        )

    def _generate_initial_hypothesis(
        self,
        case_text: str,
        task_type: str,
        options: Optional[List[str]] = None
    ) -> Tuple[str, float]:
        """
        Generate initial hypothesis using Pass 1 prompt.

        Args:
            case_text: Partial case text (20% tokens)
            task_type: "mcq" or "free_text"
            options: MCQ options if applicable

        Returns:
            Tuple of (hypothesis, confidence)
        """
        prompt = PreMortemPrompts.build_initial_hypothesis_prompt(
            case_text, task_type, options
        )

        response = self._call_llm(prompt)
        hypothesis = response.strip()

        # Default confidence for Pass 1 (partial information)
        confidence = 0.7

        return hypothesis, confidence

    def _run_premortem(
        self,
        case_text: str,
        hypothesis: str
    ) -> PreMortemResponse:
        """
        Run Pre-Mortem analysis on the initial hypothesis.

        Args:
            case_text: Partial case text
            hypothesis: Initial hypothesis to challenge

        Returns:
            Parsed PreMortemResponse
        """
        prompt = PreMortemPrompts.build_premortem_prompt(
            case_text=case_text,
            hypothesis=hypothesis,
            token_percentage=self.config.initial_token_percentage * 100
        )

        response = self._call_llm(prompt)
        return PreMortemPrompts.parse_premortem_response(response)

    def _generate_final_diagnosis(
        self,
        full_case: str,
        initial_hypothesis: str,
        initial_confidence: float,
        premortem_result: Optional[PreMortemResponse],
        task_type: str,
        options: Optional[List[str]] = None
    ) -> Tuple[str, float]:
        """
        Generate final diagnosis with full case and optional Pre-Mortem context.

        Args:
            full_case: Complete case text
            initial_hypothesis: Hypothesis from Pass 1
            initial_confidence: Confidence from Pass 1
            premortem_result: Pre-Mortem result if applied
            task_type: "mcq" or "free_text"
            options: MCQ options if applicable

        Returns:
            Tuple of (diagnosis, confidence)
        """
        if premortem_result:
            # With Pre-Mortem: use belief revision prompt
            prompt = PreMortemPrompts.build_belief_revision_prompt(
                full_case=full_case,
                initial_hypothesis=initial_hypothesis,
                premortem_alternative=premortem_result.alternative_diagnosis,
                evidence_strength=premortem_result.evidence_strength,
                premortem_recommendation=premortem_result.recommendation,
                task_type=task_type,
                options=options,
                token_percentage=self.config.initial_token_percentage * 100
            )
        else:
            # Without Pre-Mortem: standard prompt with full case
            prompt = PreMortemPrompts.build_final_diagnosis_prompt(
                full_case, task_type, options
            )

        response = self._call_llm(prompt)
        diagnosis = response.strip()

        # Default confidence for Pass 2 (full information)
        confidence = 0.8

        # Adjust confidence if Pre-Mortem suggested ESCALATE
        if premortem_result and premortem_result.recommendation == "ESCALATE":
            confidence = min(confidence, 0.6)

        return diagnosis, confidence

    def _diagnoses_differ(self, hypothesis: str, diagnosis: str) -> bool:
        """
        Check if the initial hypothesis differs from final diagnosis.

        Handles both MCQ (number comparison) and free-text cases.

        Args:
            hypothesis: Initial hypothesis
            diagnosis: Final diagnosis

        Returns:
            True if they differ
        """
        # Normalize for comparison
        hyp_normalized = hypothesis.strip().lower()
        diag_normalized = diagnosis.strip().lower()

        # For MCQ, compare first character (the number)
        if hyp_normalized and diag_normalized:
            if hyp_normalized[0].isdigit() and diag_normalized[0].isdigit():
                return hyp_normalized[0] != diag_normalized[0]

        # For free-text, compare the full strings
        return hyp_normalized != diag_normalized

    def _call_llm(self, prompt: str) -> str:
        """
        Call the LLM with retry logic.

        Args:
            prompt: The prompt to send

        Returns:
            Model response text
        """
        max_retries = 3

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.config.temperature
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                if self.config.verbose:
                    print(f"[LLM] Attempt {attempt + 1} failed: {e}")

                if attempt < max_retries - 1:
                    wait_time = 60 * (attempt + 1)
                    time.sleep(wait_time)
                else:
                    raise e

        return ""  # Should never reach here


def evaluate_case_with_premortem(
    llm_client: Any,
    case_text_20pct: str,
    case_text_full: str,
    task_type: str = "free_text",
    options: Optional[List[str]] = None,
    config: Optional[PreMortemConfig] = None
) -> DiagnosisResult:
    """
    Convenience function to evaluate a case without instantiating the engine.

    Args:
        llm_client: OpenAI client instance
        case_text_20pct: Partial case text
        case_text_full: Complete case text
        task_type: "mcq" or "free_text"
        options: MCQ options if applicable
        config: PreMortemConfig instance

    Returns:
        DiagnosisResult containing evaluation information
    """
    engine = BeliefRevisionEngine(llm_client, config)
    return engine.evaluate_case(
        case_text_20pct=case_text_20pct,
        case_text_full=case_text_full,
        task_type=task_type,
        options=options
    )
