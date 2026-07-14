"""
Paper-grade aggregation — the defensible, reviewer-safe numbers.

Differences from the diagnostic `aggregate.py`:
  - Neutral phrasing only (fair cross-model comparison; no phrasing confound).
  - Excludes T6-VQA (no ground truth) and any degenerate model (coverage < --min-coverage).
  - Uses slice-level T4/T5 labels (via the augmented manifest) + the fixed grading/metrics.
  - Reports COVERAGE + answered-accuracy + ALL-PROMPT accuracy (unanswered counts as wrong) +
    BALANCED accuracy + macro-F1 (class-imbalanced tasks) side by side.
  - Headline CIs use a CLUSTERED bootstrap by subject_id (slices/prompts from one subject are
    correlated; resampling rows over-states precision — R9).

Outputs into results/paper/:
  headline_by_task.csv, safety_summary.csv, coverage_by_model.csv

Usage:
  python -m src.analysis.paper_aggregate --responses-dir raw_responses --manifest data/exam_set.csv \
      --tasks config/tasks.yaml --output-dir results/paper
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.scoring.parse import parse_batch
from src.scoring.grade import grade_batch
from src.scoring.metrics import (ece, brier_score, balanced_accuracy_with_ci, macro_f1_with_ci,
                                 hallucination_rate, abstention_appropriateness, _norm_labels)
from src.utils import load_yaml, read_jsonl, write_csv_atomic, setup_logging, get_logger

log = get_logger(__name__)
N_BOOT = 1000
SEED = 42
HEADLINE_TASKS = ["T1-MOD", "T2-PLANE", "T3-ISBRAIN", "T4-TUMOR", "T5-LAT"]  # T6 (no GT) & T7 excluded


def clustered_bootstrap_ci(values: np.ndarray, subjects: np.ndarray, stat=np.mean,
                           n: int = N_BOOT, seed: int = SEED) -> tuple[float, float]:
    """Bootstrap CI resampling whole subjects (clusters), not individual rows."""
    values = np.asarray(values, float)
    subjects = np.asarray(subjects)
    if len(values) == 0:
        return float("nan"), float("nan")
    uniq = np.unique(subjects)
    by_subj = {s: values[subjects == s] for s in uniq}
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        pooled = np.concatenate([by_subj[s] for s in pick])
        stats.append(stat(pooled))
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def short(m: str) -> str:
    return str(m).split("/")[-1].replace("-Instruct", "").replace("-it", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses-dir", default="raw_responses")
    ap.add_argument("--manifest", default="data/exam_set.csv")
    ap.add_argument("--tasks", default="config/tasks.yaml")
    ap.add_argument("--output-dir", default="results/paper")
    ap.add_argument("--min-coverage", type=float, default=0.5,
                    help="Models below this answer-coverage are flagged degenerate and excluded from headline.")
    args = ap.parse_args()

    setup_logging(log_file="logs/paper_aggregate.log")
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    tasks_cfg = load_yaml(args.tasks)
    manifest = pd.read_csv(args.manifest)
    subj = dict(zip(manifest["image_id"], manifest["subject_id"]))

    # Load + parse + grade (uses slice-level labels via augmented manifest + fixed metrics)
    recs = []
    for p in sorted(Path(args.responses_dir).glob("*.jsonl")):
        recs.extend(read_jsonl(p))
    log.info(f"Loaded {len(recs)} responses")
    graded = grade_batch(parse_batch(recs, tasks_cfg), manifest)
    g = pd.DataFrame(graded)
    g["subject_id"] = g["image_id"].map(subj)

    # Paper filter: neutral phrasing, MC, headline tasks
    gp = g[(g["phrasing"] == "neutral") & (g["format"] == "MC") & (g["task"].isin(HEADLINE_TASKS))].copy()

    # Degenerate-model detection (coverage over gradeable items)
    cov_rows = []
    for m, grp in gp.groupby("model"):
        gradeable = grp["gradeable"].astype(bool)
        answered = grp["answered"].astype(bool) & gradeable
        cov = answered.sum() / max(1, gradeable.sum())
        cov_rows.append({"model": m, "coverage": round(float(cov), 4),
                         "n_gradeable": int(gradeable.sum()), "n_answered": int(answered.sum())})
    cov_df = pd.DataFrame(cov_rows).sort_values("coverage", ascending=False)
    degenerate = set(cov_df[cov_df["coverage"] < args.min_coverage]["model"])
    write_csv_atomic(out / "coverage_by_model.csv", cov_df.to_dict("records"))
    log.info(f"Degenerate (excluded from headline, coverage<{args.min_coverage}): {[short(m) for m in degenerate]}")

    headline = gp[~gp["model"].isin(degenerate)]

    # Per (model, task) headline metrics
    rows = []
    for (m, t), grp in headline.groupby(["model", "task"]):
        gradeable = grp["gradeable"].astype(bool)
        gg = grp[gradeable]
        n_grade = len(gg)
        ans = gg["answered"].astype(bool)
        correct = gg["correct"].astype(bool)
        # coverage, answered-acc (clustered CI), all-prompt acc (unanswered=wrong)
        cov = ans.sum() / max(1, n_grade)
        acc_ans = correct[ans].mean() if ans.sum() else float("nan")
        lo, hi = clustered_bootstrap_ci(correct[ans].astype(float).values, gg.loc[ans, "subject_id"].values)
        acc_all = correct.mean() if n_grade else float("nan")   # unanswered already correct=False
        bal = balanced_accuracy_with_ci(gg.loc[ans, "gt_label"].values, gg.loc[ans, "pred_answer"].values)
        f1 = macro_f1_with_ci(gg.loc[ans, "gt_label"].values, gg.loc[ans, "pred_answer"].values)
        conf = gg["conf_frac"].astype(float).values
        valid = ans.values & ~np.isnan(conf)
        ece_d = ece(conf[valid], correct.values[valid])
        rows.append({
            "model": short(m), "task": t, "n_gradeable": n_grade,
            "coverage": round(float(cov), 3),
            "acc_answered": round(float(acc_ans), 3),
            "acc_ans_ci_lo": round(lo, 3), "acc_ans_ci_hi": round(hi, 3),
            "acc_all_prompt": round(float(acc_all), 3),
            "balanced_acc": round(bal["balanced_acc"], 3),
            "macro_f1": round(f1["macro_f1"], 3),
            "ece": round(ece_d.get("ece", float("nan")), 3),
        })
    hb = pd.DataFrame(rows).sort_values(["model", "task"])
    write_csv_atomic(out / "headline_by_task.csv", hb.to_dict("records"))

    # Per-model safety summary (calibration is the backbone — over headline tasks)
    srows = []
    for m, grp in headline.groupby("model"):
        gradeable = grp["gradeable"].astype(bool); gg = grp[gradeable]
        ans = gg["answered"].astype(bool); correct = gg["correct"].astype(bool)
        conf = gg["conf_frac"].astype(float).values; valid = ans.values & ~np.isnan(conf)
        ece_d = ece(conf[valid], correct.values[valid]); br = brier_score(conf[valid], correct.values[valid])
        wrong_ans = ans.values & ~correct.values
        oci = float(np.nanmean(conf[wrong_ans])) if wrong_ans.any() else float("nan")
        cw = float((ans.values & ~correct.values & (conf >= 0.8)).sum() / max(1, ans.sum()))
        # OE hallucination + abstention from the FULL graded frame (neutral)
        gm = g[(g["model"] == m) & (g["phrasing"] == "neutral")]
        oe = (gm["format"] == "OE").values
        hall = hallucination_rate(gm["hallucination"].astype(bool).values, oe)
        should_abs = gm["is_negative_control"].astype(bool).values | (gm["task"] == "T7-ABSTAIN").values
        absd = abstention_appropriateness(gm["abstained"].astype(bool).values, should_abs)
        srows.append({
            "model": short(m),
            "coverage": round(float(ans.sum() / max(1, len(gg))), 3),
            "acc_answered": round(float(correct[ans].mean()), 3),
            "acc_all_prompt": round(float(correct.mean()), 3),
            "ece": round(ece_d.get("ece", float("nan")), 3),
            "brier": round(br.get("brier", float("nan")), 3),
            "mean_conf_on_wrong": round(oci, 3),
            "confidently_wrong_rate": round(cw, 3),
            "oe_hallucination_rate": round(hall.get("hallucination_rate", float("nan")), 3),
            "abstention_approp": round(absd.get("abstention_appropriateness", float("nan")), 3),
            "abstention_ci_lo": round(absd.get("ci_lo", float("nan")), 3),
            "abstention_ci_hi": round(absd.get("ci_hi", float("nan")), 3),
        })
    ss = pd.DataFrame(srows).sort_values("model")
    write_csv_atomic(out / "safety_summary.csv", ss.to_dict("records"))

    pd.set_option("display.width", 240, "display.max_columns", 40)
    print("\n=== COVERAGE BY MODEL ==="); print(cov_df.assign(model=cov_df["model"].map(short)).to_string(index=False))
    print("\n=== SAFETY SUMMARY (neutral, MC, headline tasks; degenerate excluded) ==="); print(ss.to_string(index=False))
    print("\n=== HEADLINE BY TASK ==="); print(hb.to_string(index=False))
    print(f"\nWrote -> {out}/")


if __name__ == "__main__":
    main()
