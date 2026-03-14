"""
Unified LLM judge for clinical diagnosis evaluation.

Single implementation used by both plot_results.py and evaluation_with_premortem.py.
Uses DeepSeek API with strict clinical equivalence rules, retry logic, and caching.
"""

import json
import logging
import os
import re
import time
from typing import Dict, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You are a strict medical diagnosis grader. "
    "You must output ONLY valid JSON, with keys: "
    'correct (bool), confidence (0..1), rationale (string).'
)

DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"


class JudgeResult:
    """Result of a judge evaluation."""

    __slots__ = ("correct", "confidence", "rationale", "method")

    def __init__(self, correct: bool, confidence: float, rationale: str, method: str):
        self.correct = correct
        self.confidence = confidence
        self.rationale = rationale
        self.method = method

    def __repr__(self):
        return (
            f"JudgeResult(correct={self.correct!r}, confidence={self.confidence!r}, "
            f"rationale={self.rationale!r}, method={self.method!r})"
        )


class UnifiedJudge:
    """LLM-based clinical diagnosis judge with retry and caching."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "deepseek-chat",
        api_url: str = DEEPSEEK_API_URL,
        max_retries: int = 3,
        enable_cache: bool = True,
    ):
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "DEEPSEEK_API_KEY is not set. "
                "Set it as an environment variable or pass api_key to UnifiedJudge."
            )
        self.model = model
        self.api_url = api_url
        self.max_retries = max_retries
        self.enable_cache = enable_cache
        self._cache: Dict[Tuple[str, ...], JudgeResult] = {}

    def judge(self, pred: str, gold: str, case: Optional[str] = None) -> JudgeResult:
        """Judge whether pred matches gold diagnosis.

        Args:
            pred: Predicted diagnosis.
            gold: Gold/reference diagnosis.
            case: Optional clinical case text for context-aware judging.

        Returns:
            JudgeResult with correct, confidence, rationale, method.

        Raises:
            RuntimeError: If LLM judge fails after all retries.
        """
        if _is_missing(pred) or _is_missing(gold):
            return JudgeResult(
                correct=False,
                confidence=1.0,
                rationale="Missing prediction or gold diagnosis",
                method="llm",
            )

        key = self._cache_key(pred, gold, case)
        if self.enable_cache and key in self._cache:
            return self._cache[key]

        prompt = self._build_prompt(pred, gold, case)
        raw = self._call_llm(prompt)

        result = JudgeResult(
            correct=bool(raw.get("correct", False)),
            confidence=float(raw.get("confidence", 0.5)),
            rationale=str(raw.get("rationale", "")),
            method="llm",
        )

        if self.enable_cache:
            self._cache[key] = result
        return result

    def _build_prompt(self, pred: str, gold: str, case: Optional[str] = None) -> str:
        case_section = ""
        if case:
            case_section = f"CASE PRESENTATION:\n{case}\n\n"

        return (
            f"{case_section}"
            f"GOLD DIAGNOSIS: {gold}\n"
            f"PREDICTED DIAGNOSIS: {pred}\n"
            f"\n"
            f"Is the predicted diagnosis the SAME CONDITION as the gold diagnosis?\n"
            f"\n"
            f"Rules:\n"
            f'- CORRECT: Same disease/condition, even if wording differs '
            f'(e.g., "heart attack" = "myocardial infarction")\n'
            f'- CORRECT: Same core diagnosis with additional details '
            f'(e.g., "pneumonia" vs "bacterial pneumonia")\n'
            f'- INCORRECT: Different conditions, even if related '
            f'(e.g., "diabetes type 1" vs "diabetes type 2")\n'
            f"- INCORRECT: Partial match or only mentions a symptom/complication "
            f"instead of the diagnosis\n"
            f"- INCORRECT: Overly broad or vague when gold is specific\n"
            f"\n"
            f"Be STRICT. When in doubt, mark as INCORRECT.\n"
            f'Return JSON only: {{"correct": true/false, "confidence": 0.0-1.0, '
            f'"rationale": "brief reason"}}'
        )

    def _call_llm(self, prompt: str) -> dict:
        """Call DeepSeek API with retry logic. Raises on final failure."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
        }

        last_error = None
        for attempt in range(self.max_retries):
            try:
                r = requests.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=60,
                )
                r.raise_for_status()
                content = r.json()["choices"][0]["message"]["content"]
                out = _extract_json(content)
                if "correct" not in out:
                    raise ValueError(f"Judge JSON missing 'correct' key: {out}")
                out.setdefault("confidence", 0.5)
                out.setdefault("rationale", "")
                return out
            except (
                requests.RequestException,
                json.JSONDecodeError,
                KeyError,
                ValueError,
            ) as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    wait = 2**attempt  # 1s, 2s, 4s
                    logger.warning(
                        "Judge API attempt %d/%d failed: %s. Retrying in %ds...",
                        attempt + 1,
                        self.max_retries,
                        e,
                        wait,
                    )
                    time.sleep(wait)

        raise RuntimeError(
            f"LLM judge failed after {self.max_retries} attempts: {last_error}"
        ) from last_error

    @staticmethod
    def _cache_key(
        pred: str, gold: str, case: Optional[str] = None
    ) -> Tuple[str, ...]:
        p = str(pred).lower().strip()
        g = str(gold).lower().strip()
        c = str(case).lower().strip() if case else ""
        return (p, g, c)


def _extract_json(text: str) -> dict:
    """Best-effort JSON extraction from LLM output."""
    text = text.strip()
    # Strip ```json ... ``` code fences
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            raise
        return json.loads(m.group(0))


def _is_missing(val) -> bool:
    """Check if a value is NaN, None, or empty."""
    if val is None:
        return True
    # Handle float NaN
    if isinstance(val, float):
        import math
        if math.isnan(val):
            return True
    try:
        import pandas as pd
        if pd.isna(val):
            return True
    except (ImportError, TypeError, ValueError):
        pass
    if isinstance(val, str) and not val.strip():
        return True
    return False


# ---------------------------------------------------------------------------
# Backward-compatible convenience function
# ---------------------------------------------------------------------------

_default_judge: Optional[UnifiedJudge] = None


def judge_answer(
    case: str, prediction: str, gold: str, model: str = "deepseek-chat"
) -> dict:
    """Backward-compatible wrapper matching deepseek_judge.judge_answer signature.

    Returns dict with keys: correct (bool), confidence (float), rationale (str).
    """
    global _default_judge
    if _default_judge is None or _default_judge.model != model:
        _default_judge = UnifiedJudge(model=model)
    result = _default_judge.judge(pred=prediction, gold=gold, case=case)
    return {
        "correct": result.correct,
        "confidence": result.confidence,
        "rationale": result.rationale,
    }
