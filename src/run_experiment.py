
from __future__ import annotations
import argparse, json, math, random, time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .data import Problem, load_gsm8k, parse_model_answer
from .difficulty import AdaptiveTauConfig, estimate_difficulty
from .evaluate import accuracy, plot_summary
from .generate import LLM
from .prefix_consistency import SampleTrace, majority_vote, pc_wmv, truncate_prefix


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_checkpoint(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # Ignore only a possibly incomplete final line.
                continue
    return rows


def append_checkpoint(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def generate_initial_samples(llm, problem, n, max_new_tokens, temperature, top_p, do_sample):
    traces = []
    print(f"    Generating {n} initial samples...", flush=True)
    for i in range(n):
        print(f"      [initial] sample {i+1}/{n}", flush=True)
        out = llm.generate(
            problem.question,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
        )
        ans = parse_model_answer(out.text)
        print(
            f"      [initial] {i+1}/{n} done | "
            f"tokens={out.completion_tokens} | answer={ans!r}",
            flush=True,
        )
        traces.append(
            SampleTrace(
                text=out.text,
                answer=ans,
                completion_tokens=out.completion_tokens,
            )
        )
    return traces


def regenerate_at_tau(
    llm, problem, traces, tau, k_regen,
    max_new_tokens, temperature, top_p, do_sample
):
    answers = []
    token_cost = 0

    print(
        f"    [regen] tau={tau:.3f} | "
        f"{len(traces)} traces | k={k_regen}",
        flush=True,
    )

    for j, tr in enumerate(traces):
        prefix = truncate_prefix(tr.text, tau)
        local = []

        for _ in range(k_regen):
            regen_budget = max(
                64,
                int(max_new_tokens * (1.0 - tau) + 64),
            )
            out = llm.generate(
                problem.question,
                prefix=prefix,
                max_new_tokens=regen_budget,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
            )
            local.append(parse_model_answer(out.text))
            token_cost += out.completion_tokens

        ans = majority_vote(local) if local else ""
        answers.append(ans)

        print(
            f"      [regen tau={tau:.3f}] "
            f"{j+1}/{len(traces)} done | answer={ans!r}",
            flush=True,
        )

    return answers, token_cost


def run_one_problem(llm, problem, cfg, adaptive_cfg):
    n = int(cfg["n_samples"])
    k_regen = int(cfg.get("k_regen", 1))
    max_new_tokens = int(cfg["max_new_tokens"])
    temperature = float(cfg["temperature"])
    top_p = float(cfg["top_p"])
    do_sample = bool(cfg["do_sample"])
    fixed_tau = float(cfg.get("fixed_tau", 0.75))
    n_pilot = int(cfg.get("n_pilot", min(4, n)))
    weight_name = str(cfg.get("pc_weight", "cubic"))

    traces = generate_initial_samples(
        llm, problem, n, max_new_tokens,
        temperature, top_p, do_sample
    )

    originals = [t.answer for t in traces]
    init_tokens = sum(t.completion_tokens for t in traces)

    mv_pred = majority_vote(originals)

    pilot = originals[:max(1, min(n_pilot, len(originals)))]
    diff = estimate_difficulty(pilot, adaptive_cfg)
    adaptive_tau = float(diff.tau)

    print(
        f"    [difficulty] agreement={diff.agreement:.3f} "
        f"entropy={diff.entropy:.3f} label={diff.label} "
        f"adaptive_tau={adaptive_tau:.3f}",
        flush=True,
    )

    fixed_regens, fixed_tokens = regenerate_at_tau(
        llm, problem, traces, fixed_tau, k_regen,
        max_new_tokens, temperature, top_p, do_sample
    )

    pc_fixed_pred, fixed_scores = pc_wmv(
        originals, fixed_regens,
        weight_name=weight_name,
        k=k_regen,
    )

    if math.isclose(adaptive_tau, fixed_tau, abs_tol=1e-9):
        print(
            "    [adaptive] tau equals fixed tau; "
            "reusing fixed regeneration.",
            flush=True,
        )
        adaptive_regens = fixed_regens
        adaptive_tokens = 0
    else:
        adaptive_regens, adaptive_tokens = regenerate_at_tau(
            llm, problem, traces, adaptive_tau, k_regen,
            max_new_tokens, temperature, top_p, do_sample
        )

    pc_adaptive_pred, adaptive_scores = pc_wmv(
        originals, adaptive_regens,
        weight_name=weight_name,
        k=k_regen,
    )

    gold = problem.gold

    return {
        "index": problem.index,
        "question": problem.question,
        "gold": gold,
        "mv_pred": mv_pred,
        "pc_fixed_pred": pc_fixed_pred,
        "pc_adaptive_pred": pc_adaptive_pred,
        "mv_correct": int(mv_pred == gold and mv_pred != ""),
        "pc_fixed_correct": int(pc_fixed_pred == gold and pc_fixed_pred != ""),
        "pc_adaptive_correct": int(pc_adaptive_pred == gold and pc_adaptive_pred != ""),
        "n_samples": n,
        "k_regen": k_regen,
        "fixed_tau": fixed_tau,
        "adaptive_tau": adaptive_tau,
        "difficulty_label": diff.label,
        "pilot_agreement": diff.agreement,
        "pilot_entropy": diff.entropy,
        "pilot_majority": diff.majority,
        "n_pilot": diff.n_pilot,
        "original_answers": originals,
        "fixed_regens": fixed_regens,
        "adaptive_regens": adaptive_regens,
        "init_tokens": init_tokens,
        "mv_tokens": init_tokens,
        "pc_fixed_regen_tokens": fixed_tokens,
        "pc_adaptive_regen_tokens": adaptive_tokens,
        "pc_fixed_tokens": init_tokens + fixed_tokens,
        "pc_adaptive_tokens": init_tokens + adaptive_tokens,
        "pc_fixed_scores": fixed_scores,
        "pc_adaptive_scores": adaptive_scores,
        "pass_at_1": sum(a == gold for a in originals) / max(1, len(originals)),
    }


def make_final_outputs(rows, out_dir):
    import pandas as pd
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "results.csv", index=False)

    golds = df["gold"].tolist()
    methods = [
        ("mv_pred", "Majority Vote", "mv_tokens"),
        ("pc_fixed_pred", "PC-WMV fixed (tau=0.75)", "pc_fixed_tokens"),
        ("pc_adaptive_pred", "PC-WMV adaptive", "pc_adaptive_tokens"),
    ]

    summary = []
    for pred_col, name, token_col in methods:
        preds = df[pred_col].tolist()
        summary.append({
            "method": name,
            "accuracy": accuracy(preds, golds),
            "correct": int(sum(p == g and p != "" for p, g in zip(preds, golds))),
            "total": len(df),
            "mean_tokens": float(df[token_col].mean()),
        })

    pd.DataFrame(summary).to_csv(out_dir / "summary.csv", index=False)

    paired = df[
        ["index", "mv_correct", "pc_fixed_correct", "pc_adaptive_correct"]
    ].copy()
    paired["fixed_minus_mv"] = paired["pc_fixed_correct"] - paired["mv_correct"]
    paired["adaptive_minus_mv"] = paired["pc_adaptive_correct"] - paired["mv_correct"]
    paired["adaptive_minus_fixed"] = paired["pc_adaptive_correct"] - paired["pc_fixed_correct"]
    paired.to_csv(out_dir / "paired_outcomes.csv", index=False)

    difficulty = (
        df.groupby("difficulty_label")
        .agg(
            n=("index", "size"),
            mv_accuracy=("mv_correct", "mean"),
            pc_fixed_accuracy=("pc_fixed_correct", "mean"),
            pc_adaptive_accuracy=("pc_adaptive_correct", "mean"),
            mean_pilot_agreement=("pilot_agreement", "mean"),
            mean_adaptive_tau=("adaptive_tau", "mean"),
        )
        .reset_index()
    )
    difficulty.to_csv(out_dir / "difficulty_analysis.csv", index=False)

    try:
        plot_summary(df, out_dir)
    except Exception as e:
        print(f"[warning] plot_summary failed: {e}", flush=True)

    print("\nFinal results:", flush=True)
    for r in summary:
        print(
            f"{r['method']}: accuracy={r['accuracy']:.4f} "
            f"({r['correct']}/{r['total']})",
            flush=True,
        )


def run(cfg):
    set_seed(int(cfg.get("seed", 42)))

    out_dir = Path(cfg.get("output_dir", "outputs/gsm8k_qwen06b_final"))
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(
        cfg.get(
            "checkpoint_path",
            str(out_dir / "results.jsonl"),
        )
    )

    # IMPORTANT: use the existing 95-problem checkpoint if supplied.
    existing = load_checkpoint(checkpoint_path)
    completed = {int(r["index"]) for r in existing}

    problems = load_gsm8k(
        split=str(cfg.get("split", "test")),
        num_problems=cfg.get("num_problems", 1000),
        seed=int(cfg.get("seed", 42)),
    )

    print(f"Problems requested: {len(problems)}", flush=True)
    print(f"Already completed: {len(completed)}", flush=True)
    print(f"Remaining: {len(problems) - len(completed)}", flush=True)
    print(f"Initial samples/problem: {cfg['n_samples']}", flush=True)
    print(f"Fixed tau: {cfg.get('fixed_tau', 0.75)}", flush=True)
    print("Oracle tau: DISABLED", flush=True)
    print("Tau sweep: DISABLED", flush=True)

    print(f"\nLoading model: {cfg['model_id']}", flush=True)

    llm = LLM(
        model_id=cfg["model_id"],
        dtype=str(cfg.get("dtype", "float16")),
        device_map=str(cfg.get("device_map", "auto")),
        load_in_4bit=bool(cfg.get("load_in_4bit", False)),
        enable_thinking=bool(cfg.get("enable_thinking", False)),
    )

    adaptive_cfg = AdaptiveTauConfig(**cfg.get("adaptive_tau", {}))

    start = time.time()
    new_count = 0

    for problem_no, problem in enumerate(problems, start=1):
        if problem.index in completed:
            continue

        print(f"\nPROBLEM {problem_no}/{len(problems)}", flush=True)
        print(f"Global index: {problem.index}", flush=True)

        pstart = time.time()

        row = run_one_problem(
            llm, problem, cfg, adaptive_cfg
        )

        # Persist IMMEDIATELY after the problem is complete.
        append_checkpoint(checkpoint_path, row)

        completed.add(problem.index)
        new_count += 1

        elapsed = time.time() - start
        avg = elapsed / new_count
        remaining = len(problems) - len(completed)
        eta_min = remaining * avg / 60

        print(
            f"DONE {len(completed)}/{len(problems)} | "
            f"problem_time={((time.time()-pstart)/60):.2f} min | "
            f"ETA={eta_min:.1f} min",
            flush=True,
        )

    # Reload the checkpoint so final outputs include both the old
    # 95 records and the newly completed records.
    rows = load_checkpoint(checkpoint_path)
    make_final_outputs(rows, out_dir)

    print(
        f"\nCompleted {len(rows)}/{len(problems)} problems.",
        flush=True,
    )
    print(
        f"Checkpoint: {checkpoint_path}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/gsm8k_qwen06b.yaml",
    )
    parser.add_argument("--num-problems", type=int, default=None)
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.num_problems is not None:
        cfg["num_problems"] = args.num_problems
    if args.n_samples is not None:
        cfg["n_samples"] = args.n_samples
    if args.checkpoint is not None:
        cfg["checkpoint_path"] = args.checkpoint

    # Hard-enforce the primary experiment.
    cfg["fixed_tau"] = 0.75
    cfg["n_samples"] = 16
    cfg["k_regen"] = 1

    run(cfg)


if __name__ == "__main__":
    main()