"""CPU smoke test: validates adaptive-τ logic without a GPU or model download."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import normalize_answer, parse_model_answer
from src.difficulty import AdaptiveTauConfig, estimate_difficulty
from src.prefix_consistency import (
    discrimination_gap,
    majority_vote,
    pc_wmv,
    truncate_prefix,
)


def test_parse_and_normalize() -> None:
    assert parse_model_answer("... therefore #### 42") == "42"
    assert normalize_answer("$1,234") == "1234"
    assert majority_vote(["12", "12", "7"]) == "12"


def test_truncate() -> None:
    text = "one two three four five six seven eight nine ten"
    assert truncate_prefix(text, 0.5).split() == text.split()[:5]


def test_adaptive_tau() -> None:
    cfg = AdaptiveTauConfig()
    easy = estimate_difficulty(["5", "5", "5"], cfg)
    hard = estimate_difficulty(["1", "2", "3"], cfg)
    mid = estimate_difficulty(["9", "9", "4"], cfg)
    assert easy.label == "easy" and easy.tau == cfg.tau_easy
    assert hard.label == "hard" and hard.tau == cfg.tau_hard
    assert mid.label == "medium" and mid.tau == cfg.tau_medium


def test_pc_wmv_and_gap() -> None:
    originals = ["42", "42", "7", "7"]
    regens = ["42", "42", "9", "3"]  # correct consistent; wrong inconsistent
    pred, scores = pc_wmv(originals, regens, weight_name="cubic")
    # Paper K=1 cubic: consistent → w(1)=1; inconsistent → w(0.5)=0.125 each
    assert pred == "42"
    assert abs(scores["42"] - 2.0) < 1e-9  # two consistent groups
    assert abs(scores["7"] - 0.25) < 1e-9  # two inconsistent groups contribute w(0.5) each
    assert abs(scores["9"] - 0.125) < 1e-9
    assert abs(scores["3"] - 0.125) < 1e-9
    gap = discrimination_gap("42", originals, regens)
    assert gap["r_C"] == 1.0
    assert gap["r_W"] == 0.0
    assert gap["D"] == 1.0


if __name__ == "__main__":
    test_parse_and_normalize()
    test_truncate()
    test_adaptive_tau()
    test_pc_wmv_and_gap()
    print("All smoke tests passed.")
