"""Backward-compatible shim. Real implementation in judge/unified_judge.py at project root."""

import importlib.util
import os

# Load the root-level judge.unified_judge directly by file path to avoid
# shadowing by this local judge/ package.
_unified_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'judge', 'unified_judge.py')
)
_spec = importlib.util.spec_from_file_location("judge.unified_judge", _unified_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

judge_answer = _mod.judge_answer

__all__ = ["judge_answer"]
