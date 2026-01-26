"""
Pre-Mortem Prompt Templates.

Contains prompt templates for the Pre-Mortem analysis phase and
belief revision phase, along with response parsing utilities.
"""

from typing import Optional, List
from dataclasses import dataclass


@dataclass
class PreMortemResponse:
    """
    Parsed response from Pre-Mortem analysis.

    Attributes:
        alternative_diagnosis: The most dangerous alternative diagnosis
        evidence_for_alternative: Evidence supporting the alternative
        evidence_strength: Strength of evidence (0-1)
        missed_red_flags: Red flags that might point to alternative
        recommendation: Action recommendation (MAINTAIN/REVISE/ESCALATE)
        raw_response: The original unparsed response
    """
    alternative_diagnosis: str
    evidence_for_alternative: str
    evidence_strength: float
    missed_red_flags: List[str]
    recommendation: str  # "MAINTAIN", "REVISE", or "ESCALATE"
    raw_response: str


class PreMortemPrompts:
    """Generator and parser for Pre-Mortem prompts."""

    # Main Pre-Mortem analysis prompt template
    PREMORTEM_TEMPLATE = """You are conducting a Pre-Mortem analysis on a clinical diagnosis.

CONTEXT:
- Case presentation (partial, {token_percentage:.0f}% of full case): {case_text}
- Initial hypothesis from Pass 1: {hypothesis}

TASK:
Assume your initial hypothesis is COMPLETELY WRONG. Your job is to identify what dangerous condition you might have missed.

Answer the following:

1. ALTERNATIVE DIAGNOSIS: What is the most dangerous condition that could explain this presentation if your initial hypothesis is incorrect?

2. EVIDENCE FOR ALTERNATIVE: What specific findings in the case presentation support this alternative diagnosis?

3. EVIDENCE STRENGTH: Rate the strength of evidence for the alternative diagnosis (0.0 to 1.0, where 1.0 = very strong evidence).

4. MISSED RED FLAGS: List any red flags or warning signs that might point to the alternative diagnosis.

5. RECOMMENDATION: Based on your analysis, should we:
   - MAINTAIN: Keep the original hypothesis (alternative is unlikely)
   - REVISE: Consider the alternative diagnosis seriously
   - ESCALATE: The alternative is dangerous enough to warrant immediate escalation

Format your response EXACTLY as:
ALTERNATIVE_DIAGNOSIS: [diagnosis]
EVIDENCE: [evidence]
EVIDENCE_STRENGTH: [0.0-1.0]
MISSED_RED_FLAGS: [comma-separated list or "None"]
RECOMMENDATION: [MAINTAIN/REVISE/ESCALATE]
"""

    # Belief revision prompt for final diagnosis with Pre-Mortem context
    BELIEF_REVISION_TEMPLATE = """You are completing the final diagnostic reasoning.

CONTEXT:
- Full case presentation: {full_case}
- Initial hypothesis (from {token_percentage:.0f}% of case): {initial_hypothesis}
- Pre-Mortem alternative: {premortem_alternative}
- Pre-Mortem evidence strength: {evidence_strength}
- Pre-Mortem recommendation: {premortem_recommendation}

TASK:
Based on the FULL case (not just the initial partial information), provide your final diagnosis.

Consider:
1. Does the full case support your initial hypothesis?
2. Does new information strengthen or weaken the Pre-Mortem alternative?
3. What information changed your confidence?

{task_specific_instruction}

BELIEF REVISION LOG:
- Initial confidence in hypothesis: [percentage]
- Final confidence after full case: [percentage]
- Key information that changed assessment: [brief explanation]

FINAL ANSWER: {answer_format}
"""

    # Simple prompt for initial hypothesis generation
    INITIAL_HYPOTHESIS_MCQ_TEMPLATE = """Predict the diagnosis of this case presentation (Initial assessment with partial information).
Return only the correct index from the following list, for example: 3

{options_text}

Case presentation: {case_text}"""

    INITIAL_HYPOTHESIS_FREE_TEXT_TEMPLATE = """Predict the diagnosis of this case presentation (Initial assessment with partial information).
Return the final diagnosis in one concise sentence without any further elaboration.
For example: <diagnosis name here>

Case presentation: {case_text}

Diagnosis:"""

    # Final diagnosis prompts (without Pre-Mortem)
    FINAL_DIAGNOSIS_MCQ_TEMPLATE = """Predict the diagnosis of this case presentation (Full case review).
Return only the correct index from the following list, for example: 3

{options_text}

Case presentation: {case_text}"""

    FINAL_DIAGNOSIS_FREE_TEXT_TEMPLATE = """Predict the diagnosis of this case presentation (Full case review).
Return the final diagnosis in one concise sentence without any further elaboration.
For example: <diagnosis name here>

Case presentation: {case_text}

Diagnosis:"""

    @staticmethod
    def build_premortem_prompt(
        case_text: str,
        hypothesis: str,
        token_percentage: float = 20.0
    ) -> str:
        """
        Build the Pre-Mortem analysis prompt.

        Args:
            case_text: The partial case presentation text
            hypothesis: The initial hypothesis to challenge
            token_percentage: Percentage of full case used (for context)

        Returns:
            Formatted Pre-Mortem prompt string
        """
        return PreMortemPrompts.PREMORTEM_TEMPLATE.format(
            case_text=case_text,
            hypothesis=hypothesis,
            token_percentage=token_percentage
        )

    @staticmethod
    def build_belief_revision_prompt(
        full_case: str,
        initial_hypothesis: str,
        premortem_alternative: str,
        evidence_strength: float,
        premortem_recommendation: str,
        task_type: str = "mcq",
        options: Optional[List[str]] = None,
        token_percentage: float = 20.0
    ) -> str:
        """
        Build the belief revision prompt for final diagnosis.

        Args:
            full_case: The complete case presentation
            initial_hypothesis: Initial diagnosis from Pass 1
            premortem_alternative: Alternative from Pre-Mortem analysis
            evidence_strength: Evidence strength from Pre-Mortem
            premortem_recommendation: Recommendation from Pre-Mortem
            task_type: "mcq" or "free_text"
            options: List of MCQ options (required if task_type is "mcq")
            token_percentage: Percentage used for initial assessment

        Returns:
            Formatted belief revision prompt string
        """
        if task_type == "mcq" and options:
            task_instruction = "Choose the best diagnosis from the options provided."
            options_text = "\n".join(
                [f"{i+1}. {opt}" for i, opt in enumerate(options)]
            )
            answer_format = (
                f"Return ONLY the number (1-{len(options)}) of the correct option.\n\n"
                f"Options:\n{options_text}"
            )
        else:
            task_instruction = "Provide your final diagnosis."
            answer_format = "Return the final diagnosis in one concise sentence."

        return PreMortemPrompts.BELIEF_REVISION_TEMPLATE.format(
            full_case=full_case,
            initial_hypothesis=initial_hypothesis,
            premortem_alternative=premortem_alternative,
            evidence_strength=evidence_strength,
            premortem_recommendation=premortem_recommendation,
            task_specific_instruction=task_instruction,
            answer_format=answer_format,
            token_percentage=token_percentage
        )

    @staticmethod
    def build_initial_hypothesis_prompt(
        case_text: str,
        task_type: str = "free_text",
        options: Optional[List[str]] = None
    ) -> str:
        """
        Build prompt for initial hypothesis generation (Pass 1).

        Args:
            case_text: The partial case presentation
            task_type: "mcq" or "free_text"
            options: List of MCQ options (required if task_type is "mcq")

        Returns:
            Formatted prompt string
        """
        if task_type == "mcq" and options:
            options_text = "\n".join(
                [f"{i+1}. {opt}" for i, opt in enumerate(options)]
            )
            return PreMortemPrompts.INITIAL_HYPOTHESIS_MCQ_TEMPLATE.format(
                case_text=case_text,
                options_text=options_text
            )
        else:
            return PreMortemPrompts.INITIAL_HYPOTHESIS_FREE_TEXT_TEMPLATE.format(
                case_text=case_text
            )

    @staticmethod
    def build_final_diagnosis_prompt(
        case_text: str,
        task_type: str = "free_text",
        options: Optional[List[str]] = None
    ) -> str:
        """
        Build prompt for final diagnosis (without Pre-Mortem context).

        Args:
            case_text: The full case presentation
            task_type: "mcq" or "free_text"
            options: List of MCQ options (required if task_type is "mcq")

        Returns:
            Formatted prompt string
        """
        if task_type == "mcq" and options:
            options_text = "\n".join(
                [f"{i+1}. {opt}" for i, opt in enumerate(options)]
            )
            return PreMortemPrompts.FINAL_DIAGNOSIS_MCQ_TEMPLATE.format(
                case_text=case_text,
                options_text=options_text
            )
        else:
            return PreMortemPrompts.FINAL_DIAGNOSIS_FREE_TEXT_TEMPLATE.format(
                case_text=case_text
            )

    @staticmethod
    def parse_premortem_response(response: str) -> PreMortemResponse:
        """
        Parse the structured response from Pre-Mortem analysis.

        Expects response in format:
        ALTERNATIVE_DIAGNOSIS: [diagnosis]
        EVIDENCE: [evidence]
        EVIDENCE_STRENGTH: [0.0-1.0]
        MISSED_RED_FLAGS: [comma-separated list or "None"]
        RECOMMENDATION: [MAINTAIN/REVISE/ESCALATE]

        Args:
            response: Raw response string from the model

        Returns:
            Parsed PreMortemResponse object
        """
        lines = response.strip().split('\n')

        # Default values
        result = {
            'alternative_diagnosis': '',
            'evidence_for_alternative': '',
            'evidence_strength': 0.5,
            'missed_red_flags': [],
            'recommendation': 'MAINTAIN'
        }

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith('ALTERNATIVE_DIAGNOSIS:'):
                result['alternative_diagnosis'] = line.split(':', 1)[1].strip()

            elif line.startswith('EVIDENCE:'):
                result['evidence_for_alternative'] = line.split(':', 1)[1].strip()

            elif line.startswith('EVIDENCE_STRENGTH:'):
                try:
                    value = line.split(':', 1)[1].strip()
                    # Handle cases like "0.7 (moderate)" by taking first number
                    value = value.split()[0] if ' ' in value else value
                    result['evidence_strength'] = float(value)
                    # Clamp to valid range
                    result['evidence_strength'] = max(0.0, min(1.0, result['evidence_strength']))
                except (ValueError, IndexError):
                    result['evidence_strength'] = 0.5

            elif line.startswith('MISSED_RED_FLAGS:'):
                flags = line.split(':', 1)[1].strip()
                if flags.lower() not in ('none', 'n/a', '-', ''):
                    result['missed_red_flags'] = [
                        f.strip() for f in flags.split(',') if f.strip()
                    ]

            elif line.startswith('RECOMMENDATION:'):
                rec = line.split(':', 1)[1].strip().upper()
                # Extract just the recommendation word
                for valid_rec in ['MAINTAIN', 'REVISE', 'ESCALATE']:
                    if valid_rec in rec:
                        result['recommendation'] = valid_rec
                        break

        return PreMortemResponse(
            alternative_diagnosis=result['alternative_diagnosis'],
            evidence_for_alternative=result['evidence_for_alternative'],
            evidence_strength=result['evidence_strength'],
            missed_red_flags=result['missed_red_flags'],
            recommendation=result['recommendation'],
            raw_response=response
        )
