"""Tests for the unified judge module.

Runnable with: python3 -m unittest tests.test_judge -v
Also works with pytest if available.
"""

import json
import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# Ensure project root is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from judge.unified_judge import (
    JudgeResult,
    UnifiedJudge,
    _extract_json,
    _is_missing,
    judge_answer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_api_response(correct=True, confidence=0.9, rationale="Same condition"):
    """Create a mock requests.Response for DeepSeek API."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "correct": correct,
                            "confidence": confidence,
                            "rationale": rationale,
                        }
                    )
                }
            }
        ]
    }
    return mock_resp


# ---------------------------------------------------------------------------
# JudgeResult
# ---------------------------------------------------------------------------

class TestJudgeResult(unittest.TestCase):
    def test_fields(self):
        r = JudgeResult(correct=True, confidence=0.95, rationale="match", method="llm")
        self.assertTrue(r.correct)
        self.assertEqual(r.confidence, 0.95)
        self.assertEqual(r.rationale, "match")
        self.assertEqual(r.method, "llm")


# ---------------------------------------------------------------------------
# _extract_json
# ---------------------------------------------------------------------------

class TestExtractJson(unittest.TestCase):
    def test_clean_json(self):
        text = '{"correct": true, "confidence": 0.9, "rationale": "ok"}'
        result = _extract_json(text)
        self.assertTrue(result["correct"])

    def test_code_fenced_json(self):
        text = '```json\n{"correct": false, "confidence": 0.1, "rationale": "no"}\n```'
        result = _extract_json(text)
        self.assertFalse(result["correct"])

    def test_embedded_json(self):
        text = 'Here is the result: {"correct": true, "confidence": 0.8, "rationale": "yes"} done.'
        result = _extract_json(text)
        self.assertTrue(result["correct"])

    def test_invalid_json_raises(self):
        with self.assertRaises((json.JSONDecodeError, ValueError)):
            _extract_json("no json here at all")


# ---------------------------------------------------------------------------
# _is_missing
# ---------------------------------------------------------------------------

class TestIsMissing(unittest.TestCase):
    def test_none(self):
        self.assertTrue(_is_missing(None))

    def test_empty_string(self):
        self.assertTrue(_is_missing(""))
        self.assertTrue(_is_missing("   "))

    def test_valid_string(self):
        self.assertFalse(_is_missing("pneumonia"))

    def test_nan_float(self):
        self.assertTrue(_is_missing(float("nan")))


# ---------------------------------------------------------------------------
# UnifiedJudge._build_prompt
# ---------------------------------------------------------------------------

class TestBuildPrompt(unittest.TestCase):
    def setUp(self):
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}):
            self.judge = UnifiedJudge()

    def test_without_case(self):
        prompt = self.judge._build_prompt("pneumonia", "bacterial pneumonia")
        self.assertIn("GOLD DIAGNOSIS: bacterial pneumonia", prompt)
        self.assertIn("PREDICTED DIAGNOSIS: pneumonia", prompt)
        self.assertNotIn("CASE PRESENTATION", prompt)
        self.assertIn("When in doubt, mark as INCORRECT", prompt)

    def test_with_case(self):
        prompt = self.judge._build_prompt(
            "pneumonia", "bacterial pneumonia", case="Patient with cough and fever"
        )
        self.assertIn("CASE PRESENTATION:", prompt)
        self.assertIn("Patient with cough and fever", prompt)
        self.assertIn("GOLD DIAGNOSIS: bacterial pneumonia", prompt)


# ---------------------------------------------------------------------------
# UnifiedJudge.judge
# ---------------------------------------------------------------------------

class TestJudge(unittest.TestCase):
    def setUp(self):
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}):
            self.judge = UnifiedJudge(enable_cache=True)

    @patch("judge.unified_judge.requests.post")
    def test_successful_call(self, mock_post):
        mock_post.return_value = _mock_api_response(correct=True, confidence=0.95)
        result = self.judge.judge("pneumonia", "bacterial pneumonia")
        self.assertTrue(result.correct)
        self.assertEqual(result.confidence, 0.95)
        self.assertEqual(result.method, "llm")

    @patch("judge.unified_judge.requests.post")
    def test_incorrect_diagnosis(self, mock_post):
        mock_post.return_value = _mock_api_response(
            correct=False, confidence=0.9, rationale="Different conditions"
        )
        result = self.judge.judge("diabetes type 1", "diabetes type 2")
        self.assertFalse(result.correct)

    @patch("judge.unified_judge.requests.post")
    def test_missing_pred_returns_false(self, mock_post):
        result = self.judge.judge(None, "pneumonia")
        self.assertFalse(result.correct)
        self.assertIn("Missing", result.rationale)
        mock_post.assert_not_called()

    @patch("judge.unified_judge.requests.post")
    def test_missing_gold_returns_false(self, mock_post):
        result = self.judge.judge("pneumonia", "")
        self.assertFalse(result.correct)
        mock_post.assert_not_called()

    @patch("judge.unified_judge.requests.post")
    def test_cache_hit(self, mock_post):
        mock_post.return_value = _mock_api_response(correct=True)
        r1 = self.judge.judge("pneumonia", "pneumonia")
        r2 = self.judge.judge("pneumonia", "pneumonia")
        self.assertEqual(r1.correct, r2.correct)
        self.assertEqual(mock_post.call_count, 1)

    @patch("judge.unified_judge.requests.post")
    def test_cache_miss_different_inputs(self, mock_post):
        mock_post.return_value = _mock_api_response(correct=True)
        self.judge.judge("pneumonia", "pneumonia")
        self.judge.judge("diabetes", "diabetes")
        self.assertEqual(mock_post.call_count, 2)

    @patch("judge.unified_judge.requests.post")
    def test_cache_disabled(self, mock_post):
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}):
            judge_no_cache = UnifiedJudge(enable_cache=False)
        mock_post.return_value = _mock_api_response(correct=True)
        judge_no_cache.judge("pneumonia", "pneumonia")
        judge_no_cache.judge("pneumonia", "pneumonia")
        self.assertEqual(mock_post.call_count, 2)

    @patch("judge.unified_judge.requests.post")
    @patch("judge.unified_judge.time.sleep")
    def test_retry_then_succeed(self, mock_sleep, mock_post):
        """First call fails, second succeeds."""
        import requests as req

        mock_post.side_effect = [
            req.exceptions.ConnectionError("timeout"),
            _mock_api_response(correct=True),
        ]
        result = self.judge.judge("pneumonia_retry", "pneumonia_retry")
        self.assertTrue(result.correct)
        self.assertEqual(mock_post.call_count, 2)
        mock_sleep.assert_called_once_with(1)  # 2^0 = 1s backoff

    @patch("judge.unified_judge.requests.post")
    @patch("judge.unified_judge.time.sleep")
    def test_all_retries_fail_raises(self, mock_sleep, mock_post):
        """All retries fail — should raise RuntimeError."""
        import requests as req

        mock_post.side_effect = req.exceptions.ConnectionError("down")
        with self.assertRaises(RuntimeError):
            self.judge.judge("fail_test", "fail_test")
        self.assertEqual(mock_post.call_count, 3)

    @patch("judge.unified_judge.requests.post")
    def test_with_case_context(self, mock_post):
        mock_post.return_value = _mock_api_response(correct=True)
        result = self.judge.judge("pneumonia_ctx", "pneumonia_ctx", case="Patient with fever")
        self.assertTrue(result.correct)
        call_payload = mock_post.call_args[1]["json"]
        user_msg = call_payload["messages"][1]["content"]
        self.assertIn("CASE PRESENTATION:", user_msg)
        self.assertIn("Patient with fever", user_msg)


# ---------------------------------------------------------------------------
# judge_answer (backward-compat wrapper)
# ---------------------------------------------------------------------------

class TestJudgeAnswerWrapper(unittest.TestCase):
    @patch("judge.unified_judge.requests.post")
    def test_returns_dict(self, mock_post):
        mock_post.return_value = _mock_api_response(
            correct=True, confidence=0.9, rationale="match"
        )
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}):
            import judge.unified_judge as mod
            mod._default_judge = None

            out = judge_answer(
                case="Patient with cough",
                prediction="pneumonia",
                gold="pneumonia",
            )
        self.assertIsInstance(out, dict)
        self.assertIn("correct", out)
        self.assertIn("confidence", out)
        self.assertIn("rationale", out)
        self.assertTrue(out["correct"])


if __name__ == "__main__":
    unittest.main()
