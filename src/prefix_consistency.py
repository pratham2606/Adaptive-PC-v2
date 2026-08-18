"""Prefix consistency scoring and PC-WMV aggregation."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Callable, Iterable

from .data import normalize_answer, parse_model_answer


WeightFn = Callable[[float], float]


def weight_linear(c: float) -> float:
    return c


def weight_quadratic(c: float) -> float:
    return c * c


def weight_cubic(c: float) -> float:
    return c * c * c


WEIGHTS: dict[str, WeightFn] = {
    "linear": weight_linear,
    "quadratic": weight_quadratic,
    "cubic": weight_cubic,
}


def truncate_prefix(text: str, tau: float) -> str:
    """Truncate a CoT string at fraction tau of its whitespace tokens."""
    if not text:
        return ""
    tau = min(max(tau, 0.05), 0.95)
    tokens = text.split()
    if len(tokens) <= 1:
        return text
    cut = max(1, int(round(tau * len(tokens))))
    cut = min(cut, len(tokens) - 1)
    return " ".join(tokens[:cut])


@dataclass
class SampleTrace:
    text: str
    answer: str
    completion_tokens: int = 0


@dataclass
class PCScore:
    original_answer: str
    regenerated_answer: str
    consistent: bool
    tau: float
    prefix: str
    regen_text: str
    regen_tokens: int


def prefix_consistency(
    original: SampleTrace,
    regenerated: SampleTrace,
    tau: float,
    prefix: str,
) -> PCScore:
    orig = normalize_answer(original.answer)
    regen = normalize_answer(regenerated.answer)
    return PCScore(
        original_answer=orig,
        regenerated_answer=regen,
        consistent=(orig != "" and orig == regen),
        tau=tau,
        prefix=prefix,
        regen_text=regenerated.text,
        regen_tokens=regenerated.completion_tokens,
    )


def majority_vote(answers: Iterable[str]) -> str:
    cleaned = [normalize_answer(a) for a in answers if normalize_answer(a)]
    if not cleaned:
        return ""
    return Counter(cleaned).most_common(1)[0][0]


def pc_wmv(
    original_answers: list[str],
    regenerated_answers: list[str],
    weight_name: str = "cubic",
    k: int = 1,
) -> tuple[str, dict[str, float]]:
    """
    Paper-faithful PC-WMV (Iwase et al., 2026), Eqs. (7)–(8) / Algorithm 1.

    For each group i with answers A_i = {a_i, ã_i,1, ..., ã_i,K}:
        c_i(a) = |{a' in A_i : a' = a}| / (K + 1)
        votes[a] += w(c_i(a))   for each distinct a in A_i

    With K=1:
      - consistent (a_i = ã_i): c=1  → votes[a] += w(1)
      - inconsistent:           c=1/2 for each → votes[a_i] += w(0.5),
                                                 votes[ã_i] += w(0.5)
    """
    w = WEIGHTS[weight_name]
    scores: dict[str, float] = defaultdict(float)

    # regenerated_answers may be one regen per group (K=1) or a flat list;
    # we treat zip as one regenerated answer per original (K=1 default).
    for orig, regen in zip(original_answers, regenerated_answers):
        group = [normalize_answer(orig), normalize_answer(regen)]
        group = [a for a in group if a]
        if not group:
            continue
        denom = float(k + 1)
        # Count multiplicity in this group (length may be < K+1 if parse failed).
        counts = Counter(group)
        for a, cnt in counts.items():
            c = cnt / denom
            scores[a] += w(c)

    if not scores:
        return "", {}
    best = max(scores.items(), key=lambda kv: kv[1])[0]
    return best, dict(scores)


def discrimination_gap(
    gold: str,
    originals: list[str],
    regenerations: list[str],
) -> dict[str, float]:
    """Estimate r_C, r_W, D on one problem."""
    gold = normalize_answer(gold)
    correct_flags = []
    consistent_flags = []
    for o, r in zip(originals, regenerations):
        o_n = normalize_answer(o)
        r_n = normalize_answer(r)
        correct_flags.append(o_n == gold and o_n != "")
        consistent_flags.append(o_n != "" and o_n == r_n)

    correct_idx = [i for i, ok in enumerate(correct_flags) if ok]
    wrong_idx = [i for i, ok in enumerate(correct_flags) if not ok]

    r_c = (
        sum(consistent_flags[i] for i in correct_idx) / len(correct_idx)
        if correct_idx
        else float("nan")
    )
    r_w = (
        sum(consistent_flags[i] for i in wrong_idx) / len(wrong_idx)
        if wrong_idx
        else float("nan")
    )
    d = r_c - r_w if correct_idx and wrong_idx else float("nan")
    return {
        "r_C": r_c,
        "r_W": r_w,
        "D": d,
        "n_correct": float(len(correct_idx)),
        "n_wrong": float(len(wrong_idx)),
    }


def answers_from_texts(texts: list[str]) -> list[str]:
    return [parse_model_answer(t) for t in texts]
