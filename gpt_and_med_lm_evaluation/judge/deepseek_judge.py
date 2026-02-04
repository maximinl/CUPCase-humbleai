import os
import json
import re
import requests

DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

_SYSTEM = (
    "You are a strict medical answer grader. "
    "You must output ONLY valid JSON, with keys: correct (bool), confidence (0..1), rationale (string)."
)

def _extract_json(text: str):
    """Best-effort JSON extraction in case the model wraps output in text/code fences."""
    text = text.strip()
    # strip ```json ... ```
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    # try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # try to find first {...} block
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            raise
        return json.loads(m.group(0))

def judge_answer(case: str, prediction: str, gold: str, model: str = "deepseek-chat"):
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set (Colab secret not loaded).")

    prompt = f"""
CASE:
{case}

MODEL ANSWER:
{prediction}

GOLD ANSWER:
{gold}

Question: Is the model answer clinically correct with respect to the GOLD ANSWER?
Return ONLY JSON in this exact schema:
{{"correct": true/false, "confidence": 0.0-1.0, "rationale": "1-2 short sentences"}}
""".strip()

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
    }

    r = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    content = r.json()["choices"][0]["message"]["content"]
    out = _extract_json(content)

    # minimal validation
    out.setdefault("confidence", 0.5)
    out.setdefault("rationale", "")
    if "correct" not in out:
        raise ValueError(f"Judge returned JSON without 'correct': {out}")
    return out
