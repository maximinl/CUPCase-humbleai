"""
Quadrant Classifier for Risk-Based Pre-Mortem Triggering.

Implements the Dual-Reflective Trigger Map that classifies clinical cases
into risk quadrants based on complexity and stakes assessment.
"""

from typing import Optional, List
from dataclasses import dataclass

from .config import (
    PreMortemConfig,
    RiskQuadrant,
    COMPLEXITY_INDICATORS,
    STAKES_INDICATORS,
    RED_FLAG_PATTERNS,
    QUADRANT_DESCRIPTIONS
)


@dataclass
class QuadrantResult:
    """
    Result of quadrant classification.

    Attributes:
        quadrant: The assigned risk quadrant
        complexity_score: Calculated complexity score (0-1)
        stakes_score: Calculated stakes score (0-1)
        red_flags_detected: List of detected red flag patterns
        requires_premortem: Whether Pre-Mortem analysis is required
        reasoning: Human-readable explanation of the classification
    """
    quadrant: RiskQuadrant
    complexity_score: float
    stakes_score: float
    red_flags_detected: List[str]
    requires_premortem: bool
    reasoning: str


class QuadrantClassifier:
    """
    Classifies clinical cases into risk quadrants.

    The classifier analyzes case text to determine:
    - Clinical complexity (based on indicator keywords)
    - Clinical stakes (based on urgency/severity keywords)
    - Presence of red flags (patterns that always warrant attention)

    Quadrant Assignment:
    - Q1 (Routine): Low complexity, low stakes
    - Q2 (Watchful): Low complexity, high stakes
    - Q3 (Curiosity): High complexity, low stakes
    - Q4 (Escalate): High complexity, high stakes
    """

    def __init__(self, config: Optional[PreMortemConfig] = None):
        """
        Initialize the classifier.

        Args:
            config: PreMortemConfig instance. Uses defaults if not provided.
        """
        self.config = config or PreMortemConfig()

    def classify(
        self,
        case_text: str,
        hypothesis: Optional[str] = None
    ) -> QuadrantResult:
        """
        Classify a clinical case into a risk quadrant.

        Args:
            case_text: The clinical case presentation text
            hypothesis: Optional initial hypothesis from the model

        Returns:
            QuadrantResult containing classification and metrics
        """
        case_lower = case_text.lower()

        # Calculate component scores
        complexity_score = self._calculate_complexity(case_lower)
        stakes_score = self._calculate_stakes(case_lower)
        red_flags = self._detect_red_flags(case_lower)

        # Adjust stakes score if red flags are present
        if red_flags:
            # Each red flag increases stakes score
            adjustment = min(0.3, len(red_flags) * 0.1)
            stakes_score = min(1.0, stakes_score + adjustment)

        # Determine quadrant based on scores
        quadrant = self._determine_quadrant(complexity_score, stakes_score)

        # Determine if Pre-Mortem is required
        requires_premortem = self._should_trigger_premortem(
            quadrant, red_flags
        )

        # Generate reasoning explanation
        reasoning = self._generate_reasoning(
            quadrant, complexity_score, stakes_score, red_flags
        )

        return QuadrantResult(
            quadrant=quadrant,
            complexity_score=complexity_score,
            stakes_score=stakes_score,
            red_flags_detected=red_flags,
            requires_premortem=requires_premortem,
            reasoning=reasoning
        )

    def _calculate_complexity(self, text: str) -> float:
        """
        Calculate clinical complexity score.

        The score is based on:
        - Presence of high-complexity indicator words
        - Presence of low-complexity indicator words
        - Case length (longer cases tend to be more complex)

        Args:
            text: Lowercase case text

        Returns:
            Complexity score between 0 and 1
        """
        # Count indicator words
        high_count = sum(
            1 for word in COMPLEXITY_INDICATORS["high"]
            if word in text
        )
        low_count = sum(
            1 for word in COMPLEXITY_INDICATORS["low"]
            if word in text
        )

        # Calculate base score from indicator ratio
        total = high_count + low_count + 1  # +1 to avoid division by zero
        indicator_score = high_count / total

        # Adjust for case length (longer cases often more complex)
        word_count = len(text.split())
        length_factor = min(1.0, word_count / 500)

        # Combine scores with weighting
        final_score = indicator_score * 0.7 + length_factor * 0.3

        return min(1.0, final_score)

    def _calculate_stakes(self, text: str) -> float:
        """
        Calculate clinical stakes score.

        The score is based on presence of urgency and severity indicators.

        Args:
            text: Lowercase case text

        Returns:
            Stakes score between 0 and 1
        """
        # Count indicator words
        high_count = sum(
            1 for word in STAKES_INDICATORS["high"]
            if word in text
        )
        low_count = sum(
            1 for word in STAKES_INDICATORS["low"]
            if word in text
        )

        # Calculate score from indicator ratio
        total = high_count + low_count + 1
        score = high_count / total

        return min(1.0, score)

    def _detect_red_flags(self, text: str) -> List[str]:
        """
        Detect clinical red flag patterns in the text.

        Red flags are patterns that warrant heightened attention
        regardless of other complexity/stakes indicators.

        Args:
            text: Lowercase case text

        Returns:
            List of detected red flag patterns
        """
        detected = []
        for pattern in RED_FLAG_PATTERNS:
            if pattern in text:
                detected.append(pattern)
        return detected

    def _determine_quadrant(
        self,
        complexity: float,
        stakes: float
    ) -> RiskQuadrant:
        """
        Determine the risk quadrant based on complexity and stakes scores.

        Args:
            complexity: Complexity score (0-1)
            stakes: Stakes score (0-1)

        Returns:
            The appropriate RiskQuadrant
        """
        high_complexity = complexity >= self.config.complexity_threshold
        high_stakes = stakes >= self.config.stakes_threshold

        if high_complexity and high_stakes:
            return RiskQuadrant.ESCALATE
        elif high_complexity and not high_stakes:
            return RiskQuadrant.CURIOSITY
        elif not high_complexity and high_stakes:
            return RiskQuadrant.WATCHFUL
        else:
            return RiskQuadrant.ROUTINE

    def _should_trigger_premortem(
        self,
        quadrant: RiskQuadrant,
        red_flags: List[str]
    ) -> bool:
        """
        Determine if Pre-Mortem analysis should be triggered.

        Pre-Mortem is triggered if:
        - The quadrant is in the configured premortem_quadrants list, OR
        - Two or more red flags are detected

        Args:
            quadrant: The assigned risk quadrant
            red_flags: List of detected red flags

        Returns:
            True if Pre-Mortem should be triggered
        """
        # Check if quadrant triggers Pre-Mortem
        quadrant_trigger = quadrant in self.config.premortem_quadrants

        # Multiple red flags always trigger Pre-Mortem
        red_flag_trigger = len(red_flags) >= 2

        return quadrant_trigger or red_flag_trigger

    def _generate_reasoning(
        self,
        quadrant: RiskQuadrant,
        complexity: float,
        stakes: float,
        red_flags: List[str]
    ) -> str:
        """
        Generate a human-readable explanation of the classification.

        Args:
            quadrant: The assigned risk quadrant
            complexity: Complexity score
            stakes: Stakes score
            red_flags: Detected red flags

        Returns:
            Formatted reasoning string
        """
        parts = [
            f"Quadrant: {quadrant.name} ({QUADRANT_DESCRIPTIONS[quadrant]})",
            f"Complexity: {complexity:.2f} (threshold: {self.config.complexity_threshold})",
            f"Stakes: {stakes:.2f} (threshold: {self.config.stakes_threshold})"
        ]

        if red_flags:
            parts.append(f"Red flags detected: {', '.join(red_flags)}")

        return " | ".join(parts)


def classify_case(
    case_text: str,
    config: Optional[PreMortemConfig] = None
) -> QuadrantResult:
    """
    Convenience function to classify a case without instantiating the classifier.

    Args:
        case_text: The clinical case presentation text
        config: Optional PreMortemConfig

    Returns:
        QuadrantResult containing classification and metrics
    """
    classifier = QuadrantClassifier(config)
    return classifier.classify(case_text)
