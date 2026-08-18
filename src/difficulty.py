"""Difficulty estimation → adaptive truncation fraction τ."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from .data import normalize_answer
from .prefix_consistency import majority_vote


@dataclass
class AdaptiveTauConfig:
    easy_agreement: float = 0.85
    hard_agreement: float = 0.40
    tau_easy: float = 0.45
    tau_medium: float = 0.75
    tau_hard: float = 0.90


@dataclass
class DifficultyEstimate:
    agreement: float
    entropy: float
    majority: str
    label: str  # easy | medium | hard
    tau: float
    n_pilot: int


def answer_agreement(answers: list[str]) -> tuple[float, str]:
    cleaned = [normalize_answer(a) for a in answers if normalize_answer(a)]
    if not cleaned:
        return 0.0, ""
    maj = majority_vote(cleaned)
    agreement = sum(a == maj for a in cleaned) / len(cleaned)
    return agreement, maj


def answer_entropy(answers: list[str]) -> float:
    cleaned = [normalize_answer(a) for a in answers if normalize_answer(a)]
    if not cleaned:
        return 0.0
    counts = Counter(cleaned)
    total = len(cleaned)
    ent = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            ent -= p * math.log(p, 2)
    return ent


def estimate_difficulty(
    pilot_answers: list[str],
    cfg: AdaptiveTauConfig,
) -> DifficultyEstimate:
    """
    Cheap, label-free difficulty proxy.

    Intuition (aligned with Age of Verification / PC):
    - High pilot agreement  → easy → answer crystallizes early → lower τ
    - Low pilot agreement   → hard → need more of the reasoning chain → higher τ
    """
    agreement, maj = answer_agreement(pilot_answers)
    ent = answer_entropy(pilot_answers)

    if agreement >= cfg.easy_agreement:
        label, tau = "easy", cfg.tau_easy
    elif agreement <= cfg.hard_agreement:
        label, tau = "hard", cfg.tau_hard
    else:
        label, tau = "medium", cfg.tau_medium

    return DifficultyEstimate(
        agreement=agreement,
        entropy=ent,
        majority=maj,
        label=label,
        tau=tau,
        n_pilot=len(pilot_answers),
    )


def interpolate_tau(agreement: float, cfg: AdaptiveTauConfig) -> float:
    """Smooth alternative: linearly map agreement in [hard, easy] → [tau_hard, tau_easy]."""
    lo, hi = cfg.hard_agreement, cfg.easy_agreement
    if hi <= lo:
        return cfg.tau_medium
    t = (agreement - lo) / (hi - lo)
    t = min(max(t, 0.0), 1.0)
    # high agreement → low τ
    return cfg.tau_hard + t * (cfg.tau_easy - cfg.tau_hard)
