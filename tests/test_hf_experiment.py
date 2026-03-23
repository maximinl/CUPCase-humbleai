import json

import hf_experiment


class _DummyClient:
    pass


def test_ensure_diverse_candidates_repompts_for_alternatives(monkeypatch):
    calls = []

    def fake_get_distinct_differential(client, case_text, excluded, n=3):
        calls.append(tuple(excluded))
        return ["Body dysmorphic disorder"] if len(calls) == 1 else []

    monkeypatch.setattr(hf_experiment, "get_distinct_differential", fake_get_distinct_differential)

    candidates = hf_experiment.ensure_diverse_candidates(
        _DummyClient(),
        _DummyClient(),
        "case text",
        ["Bulimia nervosa", "Bulimia nervosa"],
        min_candidates=2,
    )

    assert len(candidates) == 2
    assert candidates == ["Bulimia nervosa", "Body dysmorphic disorder"]


def test_process_case_differential_mode_guarantees_multiple_candidates(monkeypatch):
    monkeypatch.setattr(hf_experiment, "get_diagnosis", lambda client, case_text: "Bulimia nervosa")
    monkeypatch.setattr(
        hf_experiment,
        "get_differential",
        lambda client, case_text, n=3: ["Bulimia nervosa", "Bulimia nervosa"],
    )
    monkeypatch.setattr(
        hf_experiment,
        "get_distinct_differential",
        lambda client, case_text, excluded, n=3: ["Body dysmorphic disorder"],
    )
    monkeypatch.setattr(
        hf_experiment,
        "perform_audit",
        lambda client, case_text, candidates, mode="legacy": {
            "final_decision": candidates[0],
            "confidence": 0.9,
            "rationale": "test",
        },
    )

    result = hf_experiment.process_case(
        _DummyClient(),
        _DummyClient(),
        "case text",
        "Gold",
        diverse_mode="differential",
    )

    candidates = json.loads(result["ensemble_candidates"])
    assert result["num_candidates"] >= 2
    assert candidates == ["Bulimia nervosa", "Body dysmorphic disorder"]
