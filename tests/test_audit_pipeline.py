import asyncio
import json
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import audit_pipeline


def _mock_response(content: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def test_differential_prompt_allows_new_final_diagnosis():
    prompt = audit_pipeline._build_differential_audit_prompt(
        "case text",
        ["Polycythemia vera", "Essential thrombocythemia"],
    )

    assert "You are NOT limited to the initial candidates" in prompt
    assert "counter_hypotheses" in prompt


def test_process_case_differential_mode_records_counter_hypotheses(monkeypatch):
    async_mock = AsyncMock(
        return_value=_mock_response(
            json.dumps(
                {
                    "counter_hypotheses": [
                        {
                            "original": "Polycythemia vera",
                            "alternatives": ["Primary myelofibrosis", "CML"],
                        }
                    ],
                    "evaluation": [
                        {
                            "diagnosis": "Primary myelofibrosis",
                            "evidence_for": ["splenomegaly"],
                            "evidence_against": [],
                            "missing_findings": [],
                        }
                    ],
                    "final_decision": "Primary myelofibrosis",
                    "confidence": 0.92,
                    "rationale": "Best fit",
                }
            )
        )
    )
    monkeypatch.setattr(
        audit_pipeline,
        "openai_client",
        SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=async_mock)
            )
        ),
    )

    row = {
        "clean text": "case text",
        "gold": "Primary myelofibrosis",
        "candidates": json.dumps(["Polycythemia vera"]),
    }
    result = asyncio.run(audit_pipeline.process_case(558, row, audit_mode="differential"))

    assert result["final_diagnosis"] == "Primary myelofibrosis"
    assert json.loads(result["counter_hypotheses"])[0]["alternatives"][0] == "Primary myelofibrosis"
