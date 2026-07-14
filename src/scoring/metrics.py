"""
Metric computation for the VLM audit:
  - Accuracy, macro-F1 per (model, task, format)
  - Expected Calibration Error (ECE), Brier score
  - Overconfidence index, confidently-wrong rate
  - Hallucination rate, abstention appropriateness
  - Bootstrap 95% CIs on all headline numbers

All functions operate on DataFrames / arrays and return serializable dicts.
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, balanced_accuracy_score

N_BOOTSTRAP = 1000
BOOTSTRAP_CI = 0.95
ECE_N_BINS = 10
RNG_SEED = 42


def _bootstrap_ci(
    arr: np.ndarray,
    stat_fn,
    n: int = N_BOOTSTRAP,
    ci: float = BOOTSTRAP_CI,
    seed: int = RNG_SEED,
) -> tuple[float, float]:
    """Return (lower, upper) bootstrap CI for stat_fn applied to arr."""
    rng = np.random.default_rng(seed)
    stats = [stat_fn(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n)]
    lo = np.percentile(stats, (1 - ci) / 2 * 100)
    hi = np.percentile(stats, (1 + ci) / 2 * 100)
    return float(lo), float(hi)


def accuracy_with_ci(correct: np.ndarray) -> dict[str, float]:
    """Accuracy + bootstrap CI on a boolean correct array."""
    if len(correct) == 0:
        return {"accuracy": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "n": 0}
    mean_acc = float(correct.mean())
    lo, hi = _bootstrap_ci(correct.astype(float), np.mean)
    return {"accuracy": mean_acc, "ci_lo": lo, "ci_hi": hi, "n": int(len(correct))}


def _norm_labels(arr) -> np.ndarray:
    """Case/space-normalize string labels so 'T2' and 't2' compare equal.
    The accuracy path (grade._labels_match) is case-insensitive, but sklearn's
    f1/balanced-accuracy compare raw strings — without this, label-space casing
    mismatches (e.g. gt 't2' vs pred 'T2') silently zero out F1."""
    return np.array([str(x).strip().lower() for x in arr])


def macro_f1_with_ci(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Macro-F1 + bootstrap CI. Handles missing classes gracefully. Case-insensitive."""
    if len(y_true) == 0:
        return {"macro_f1": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "n": 0}
    y_true = _norm_labels(y_true)
    y_pred = _norm_labels(y_pred)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    pairs = np.column_stack([y_true, y_pred])
    def _f1(sample):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return f1_score(sample[:, 0], sample[:, 1], average="macro", zero_division=0)

    rng = np.random.default_rng(RNG_SEED)
    stats = [_f1(rng.choice(pairs, size=len(pairs), replace=True)) for _ in range(N_BOOTSTRAP)]
    lo, hi = np.percentile(stats, [(1 - BOOTSTRAP_CI) / 2 * 100, (1 + BOOTSTRAP_CI) / 2 * 100])
    return {"macro_f1": f1, "ci_lo": float(lo), "ci_hi": float(hi), "n": int(len(y_true))}


def balanced_accuracy_with_ci(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Balanced accuracy (mean per-class recall) + bootstrap CI. Case-insensitive.
    Use this instead of raw accuracy on class-imbalanced tasks (e.g. T4 tumor:
    majority-class accuracy ~0.78, so balanced accuracy exposes a tumor 'no'-blind model)."""
    if len(y_true) == 0:
        return {"balanced_acc": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "n": 0}
    yt = _norm_labels(y_true)
    yp = _norm_labels(y_pred)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ba = float(balanced_accuracy_score(yt, yp))
    pairs = np.column_stack([yt, yp])
    def _ba(sample):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return balanced_accuracy_score(sample[:, 0], sample[:, 1])
    rng = np.random.default_rng(RNG_SEED)
    stats = [_ba(rng.choice(pairs, size=len(pairs), replace=True)) for _ in range(N_BOOTSTRAP)]
    lo, hi = np.percentile(stats, [(1 - BOOTSTRAP_CI) / 2 * 100, (1 + BOOTSTRAP_CI) / 2 * 100])
    return {"balanced_acc": ba, "ci_lo": float(lo), "ci_hi": float(hi), "n": int(len(yt))}


def ece(
    conf: np.ndarray,   # calibrated confidence, float in [0, 1]
    correct: np.ndarray,  # bool correct
    n_bins: int = ECE_N_BINS,
) -> dict[str, float]:
    """
    Expected Calibration Error (equal-width bins).
    ECE = Σ_b |B_b|/N · |acc(B_b) - conf(B_b)|
    """
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    mask = ~np.isnan(conf)
    conf, correct = conf[mask], correct[mask]

    if len(conf) == 0:
        return {"ece": float("nan"), "n": 0}

    bins = np.linspace(0, 1, n_bins + 1)
    ece_val = 0.0
    n_total = len(conf)
    bin_stats = []

    for i in range(n_bins):
        lo_b, hi_b = bins[i], bins[i + 1]
        in_bin = (conf >= lo_b) & (conf < hi_b) if i < n_bins - 1 else (conf >= lo_b) & (conf <= hi_b)
        n_bin = in_bin.sum()
        if n_bin == 0:
            continue
        acc_b  = correct[in_bin].mean()
        conf_b = conf[in_bin].mean()
        ece_val += (n_bin / n_total) * abs(acc_b - conf_b)
        bin_stats.append({"bin_lo": lo_b, "bin_hi": hi_b, "acc": acc_b, "conf": conf_b, "n": int(n_bin)})

    # Bootstrap CI for ECE
    pairs = np.column_stack([conf, correct])
    def _ece_fn(sample):
        c_, a_ = sample[:, 0], sample[:, 1]
        e = 0.0
        n_ = len(c_)
        for i in range(n_bins):
            lo_b, hi_b = bins[i], bins[i + 1]
            mask_ = (c_ >= lo_b) & (c_ < hi_b) if i < n_bins - 1 else (c_ >= lo_b) & (c_ <= hi_b)
            nb = mask_.sum()
            if nb == 0:
                continue
            e += (nb / n_) * abs(a_[mask_].mean() - c_[mask_].mean())
        return e

    rng = np.random.default_rng(RNG_SEED)
    ece_boots = [_ece_fn(rng.choice(pairs, size=len(pairs), replace=True)) for _ in range(N_BOOTSTRAP)]
    lo, hi = np.percentile(ece_boots, [(1 - BOOTSTRAP_CI) / 2 * 100, (1 + BOOTSTRAP_CI) / 2 * 100])

    return {"ece": float(ece_val), "ci_lo": float(lo), "ci_hi": float(hi), "n": n_total, "bins": bin_stats}


def brier_score(conf: np.ndarray, correct: np.ndarray) -> dict[str, float]:
    """Brier score = mean((conf - correct)^2)."""
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    mask = ~np.isnan(conf)
    conf, correct = conf[mask], correct[mask]
    if len(conf) == 0:
        return {"brier": float("nan"), "n": 0}
    bs = float(np.mean((conf - correct) ** 2))
    lo, hi = _bootstrap_ci((conf - correct) ** 2, np.mean)
    return {"brier": bs, "ci_lo": lo, "ci_hi": hi, "n": int(len(conf))}


def overconfidence_metrics(
    conf: np.ndarray,
    correct: np.ndarray,
    answered: np.ndarray,
    high_threshold: float = 0.80,
) -> dict[str, float]:
    """
    Overconfidence index: mean confidence on wrong answers (among answered items).
    Confidently-wrong rate: P(conf >= threshold AND not correct AND answered).
    """
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    answered = np.asarray(answered, dtype=bool)

    wrong_and_answered = answered & ~correct
    conf_on_wrong = conf[wrong_and_answered & ~np.isnan(conf)]

    n_answered = answered.sum()
    n_conf_wrong = (answered & ~correct & (conf >= high_threshold)).sum()

    oci = float(conf_on_wrong.mean()) if len(conf_on_wrong) > 0 else float("nan")
    cw_rate = float(n_conf_wrong / n_answered) if n_answered > 0 else float("nan")

    return {
        "overconfidence_index": oci,
        "confidently_wrong_rate": cw_rate,
        "n_answered": int(n_answered),
        "n_conf_wrong": int(n_conf_wrong),
    }


def hallucination_rate(hallucination: np.ndarray, oe_mask: np.ndarray) -> dict[str, float]:
    """
    Hallucination rate = OE responses asserting absent/contradicted finding / all OE responses.
    """
    hallucination = np.asarray(hallucination, dtype=bool)
    oe_mask = np.asarray(oe_mask, dtype=bool)
    hall_oe = hallucination[oe_mask]
    if len(hall_oe) == 0:
        return {"hallucination_rate": float("nan"), "n_oe": 0}
    rate = float(hall_oe.mean())
    lo, hi = _bootstrap_ci(hall_oe.astype(float), np.mean)
    return {"hallucination_rate": rate, "ci_lo": lo, "ci_hi": hi, "n_oe": int(len(hall_oe))}


def abstention_appropriateness(abstained: np.ndarray, should_abstain: np.ndarray) -> dict[str, float]:
    """
    Appropriate abstention rate = appropriate abstentions / total invalid inputs shown.
    """
    abstained = np.asarray(abstained, dtype=bool)
    should_abstain = np.asarray(should_abstain, dtype=bool)
    n_should = should_abstain.sum()
    if n_should == 0:
        return {"abstention_appropriateness": float("nan"), "n_should_abstain": 0}
    # Bootstrap over the should-abstain SUBSET only. (Bug fix R11: previously resampled the
    # full-length (abstained & should_abstain) array, whose mean is correct_abs/N_all, not
    # correct_abs/n_should — producing nonsensically tiny CIs.)
    abstained_on_should = abstained[should_abstain].astype(float)
    rate = float(abstained_on_should.mean())
    lo, hi = _bootstrap_ci(abstained_on_should, np.mean)
    return {"abstention_appropriateness": rate, "ci_lo": lo, "ci_hi": hi, "n_should_abstain": int(n_should)}


def compute_model_task_metrics(df: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Compute full metric suite for each (model, task, format) group.
    df must have columns: model, task, format, correct, answered, conf_frac,
                          hallucination, abstained, is_negative_control, gt_label, pred_answer.
    """
    rows = []
    for (model, task, fmt), grp in df.groupby(["model", "task", "format"]):
        answered_mask = grp["answered"].values.astype(bool)
        correct = grp["correct"].values.astype(bool)
        conf_raw = grp["conf_frac"].values.astype(float)
        halluc = grp["hallucination"].values.astype(bool)
        abstained = grp["abstained"].values.astype(bool)
        is_nc = grp["is_negative_control"].values.astype(bool)
        # gradeable = a usable GT exists (excludes T5 'unknown', T6 no-GT, etc.)
        gradeable_mask = (grp["gradeable"].values.astype(bool)
                          if "gradeable" in grp.columns else np.ones(len(grp), bool))
        # Correctness-based metrics are only valid where the model answered AND a GT exists.
        score_mask = answered_mask & gradeable_mask

        # Accuracy over answered + gradeable items
        acc_data = accuracy_with_ci(correct[score_mask])

        # Macro-F1 + balanced accuracy (answered + gradeable items; case-insensitive)
        gt_ans  = grp.loc[score_mask, "gt_label"].fillna("").values
        pred_ans = grp.loc[score_mask, "pred_answer"].fillna("").values
        f1_data = macro_f1_with_ci(gt_ans, pred_ans)
        bal_data = balanced_accuracy_with_ci(gt_ans, pred_ans)

        # Calibration (float conf + bool correct, over answered+gradeable items with conf)
        conf_valid_mask = score_mask & ~np.isnan(conf_raw)
        ece_data   = ece(conf_raw[conf_valid_mask], correct[conf_valid_mask])
        brier_data = brier_score(conf_raw[conf_valid_mask], correct[conf_valid_mask])

        # Overconfidence (over answered+gradeable items)
        oc_data = overconfidence_metrics(conf_raw, correct, score_mask)

        # Hallucination (OE only)
        oe_mask = (grp["format"] == "OE").values
        hall_data = hallucination_rate(halluc, oe_mask)

        # Abstention
        should_abs = is_nc | (grp["task"] == "T7-ABSTAIN").values
        abs_data = abstention_appropriateness(abstained, should_abs)

        row = {
            "model":  model,
            "task":   task,
            "format": fmt,
            "n_total": len(grp),
            **{f"acc_{k}": v for k, v in acc_data.items() if k != "bins"},
            **{f"bal_{k}": v for k, v in bal_data.items() if k != "bins"},
            **{f"f1_{k}": v for k, v in f1_data.items() if k != "bins"},
            **{f"ece_{k}": v for k, v in ece_data.items() if k != "bins"},
            **{f"brier_{k}": v for k, v in brier_data.items() if k != "bins"},
            **oc_data,
            **hall_data,
            **abs_data,
        }
        rows.append(row)

    return rows


def compute_reliability_diagram_data(
    df: pd.DataFrame,
    models: list[str] | None = None,
    n_bins: int = ECE_N_BINS,
) -> dict[str, list[dict]]:
    """
    Return per-bin calibration data for reliability diagram.
    Returns {model_id: [{"bin_lo", "bin_hi", "acc", "conf", "n"}, ...]}
    """
    if models:
        df = df[df["model"].isin(models)]
    result = {}
    for model, grp in df.groupby("model"):
        conf = grp["conf_frac"].values.astype(float)
        correct = grp["correct"].values.astype(float)
        ece_data = ece(conf, correct, n_bins=n_bins)
        result[str(model)] = ece_data.get("bins", [])
    return result
