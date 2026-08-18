"""Evaluation metrics and plotting helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import normalize_answer


def accuracy(preds: list[str], golds: list[str]) -> float:
    if not preds:
        return 0.0
    hits = sum(normalize_answer(p) == normalize_answer(g) for p, g in zip(preds, golds))
    return hits / len(preds)


def summarize_results(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    return df


def method_accuracy_table(df: pd.DataFrame) -> pd.DataFrame:
    methods = [
        "mv_pred",
        "pc_fixed_pred",
        "pc_adaptive_pred",
        "pc_oracle_pred",
    ]
    gold = df["gold"].tolist()
    out = []
    for m in methods:
        if m not in df.columns:
            continue
        tok_col = m.replace("_pred", "_tokens")
        mean_tok = float(df[tok_col].mean()) if tok_col in df.columns else np.nan
        out.append(
            {
                "method": m.replace("_pred", ""),
                "accuracy": accuracy(df[m].tolist(), gold),
                "mean_tokens": mean_tok,
            }
        )
    return pd.DataFrame(out)


def save_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def plot_summary(df: pd.DataFrame, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    table = method_accuracy_table(df)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].bar(table["method"], table["accuracy"], color="#2F6F8F")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("GSM8K — Adaptive vs Fixed PC")
    axes[0].tick_params(axis="x", rotation=20)

    if "adaptive_tau" in df.columns:
        counts = df["difficulty_label"].value_counts().reindex(
            ["easy", "medium", "hard"], fill_value=0
        )
        axes[1].bar(counts.index.astype(str), counts.values, color="#C46A3C")
        axes[1].set_title("Chosen difficulty bins")
        axes[1].set_ylabel("# problems")

    fig.tight_layout()
    path = out_dir / "summary.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_tau_gap(df: pd.DataFrame, out_dir: Path) -> Path | None:
    """If per-τ discrimination gaps exist, plot mean D(τ)."""
    cols = [c for c in df.columns if c.startswith("D_tau_")]
    if not cols:
        return None
    taus = [float(c.split("_")[-1]) for c in cols]
    means = [df[c].astype(float).mean(skipna=True) for c in cols]

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.plot(taus, means, marker="o", color="#2F6F8F")
    ax.axhline(0.0, color="gray", lw=1, ls="--")
    ax.set_xlabel("τ")
    ax.set_ylabel("mean discrimination gap D")
    ax.set_title("Does optimal τ vary? Mean D(τ)")
    fig.tight_layout()
    path = out_dir / "discrimination_gap_vs_tau.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path
