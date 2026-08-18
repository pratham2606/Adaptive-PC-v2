"""Dataset loading and answer normalization for GSM8K."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from datasets import load_dataset


ANSWER_RE = re.compile(r"####\s*(.+)")
BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")
FINAL_ANSWER_RE = re.compile(
    r"(?:final answer|the answer is|answer)\s*[:=]?\s*(-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
INTEGER_TAIL_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*$")


@dataclass
class Problem:
    index: int
    question: str
    gold: str
    raw: dict[str, Any]


def extract_gsm8k_gold(answer_field: str) -> str:
    match = ANSWER_RE.search(answer_field)
    if match:
        return normalize_answer(match.group(1))
    return normalize_answer(answer_field.strip().split("\n")[-1])


def normalize_answer(text: str) -> str:
    text = text.strip()
    text = text.replace(",", "").replace("$", "").replace("%", "")
    text = text.replace("\\", "")
    # Keep the last numeric token if present.
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    if nums:
        return nums[-1]
    return text.lower()


def strip_thinking(text: str) -> str:
    """Remove Qwen3-style <think>...</think> blocks; keep the visible answer."""
    if not text:
        return ""
    # Drop complete think blocks.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # If an unclosed think block remains, keep only content after it when possible.
    if "</think>" in text.lower():
        parts = re.split(r"</think>", text, flags=re.IGNORECASE)
        text = parts[-1]
    return text.strip()


def parse_model_answer(text: str) -> str:
    """Extract a final numeric answer from a free-form CoT string."""
    if not text:
        return ""

    text = strip_thinking(text)

    # Prefer #### style (GSM8K training format) if the model mirrors it.
    match = ANSWER_RE.search(text)
    if match:
        return normalize_answer(match.group(1))

    match = BOXED_RE.search(text)
    if match:
        return normalize_answer(match.group(1))

    match = FINAL_ANSWER_RE.search(text)
    if match:
        return normalize_answer(match.group(1))

    # Fall back to last integer-looking token in the last non-empty line.
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    for line in reversed(lines[-5:]):
        match = INTEGER_TAIL_RE.search(line.replace(",", ""))
        if match:
            return normalize_answer(match.group(1))
    return normalize_answer(text[-80:])


def load_gsm8k(split: str = "test", num_problems: int | None = 50, seed: int = 42) -> list[Problem]:
    # Prefer namespaced id (HF Hub now rejects bare "gsm8k").
    try:
        ds = load_dataset("openai/gsm8k", "main", split=split)
    except Exception:
        ds = load_dataset("gsm8k", "main", split=split)
    if num_problems is not None and num_problems < len(ds):
        ds = ds.shuffle(seed=seed).select(range(num_problems))

    problems: list[Problem] = []
    for i, row in enumerate(ds):
        problems.append(
            Problem(
                index=i,
                question=row["question"].strip(),
                gold=extract_gsm8k_gold(row["answer"]),
                raw=dict(row),
            )
        )
    return problems
