"""
Grade parsed responses against ground-truth labels.

Grading fields:
  correct                  bool  pred_answer == GT (abstained/unparseable are not counted as wrong)
  hallucination            bool  asserts positive finding contradicted by GT
  appropriate_abstention   bool  abstained on a negative control or ambiguous item
  confidently_wrong        bool  confidence >= 80 AND not correct AND not abstained
  answered                 bool  not abstained AND not unparseable
"""
from __future__ import annotations

from typing import Any

# Positive finding phrases indicating claimed pathology in OE output
_POSITIVE_FINDINGS_SET = frozenset([
    "tumor", "mass", "lesion", "neoplasm", "glioma", "glioblastoma",
    "meningioma", "metastasis", "metastases", "hemorrhage", "bleed",
    "hematoma", "infarct", "infarction", "stroke", "edema", "enhancement",
    "ring-enhancing", "mass effect", "midline shift", "abscess",
    "signal abnormality", "hyperintense lesion", "hypointense lesion",
    "restricted diffusion", "t2 hyperintensity", "perilesional edema",
    "vasogenic edema", "white matter changes", "contrast enhancement",
])

CONFIDENCE_HIGH_THRESHOLD = 80     # "confidently wrong" cutoff
ABSTAIN_TOKENS = {"UNSURE", "UNPARSEABLE"}


def is_hallucination(
    asserted_findings: list[str],
    label_tumor_present: str,
    pred_answer: str,
    label_laterality: str,
    pred_answer_lat: str | None,
    task: str,
    format_: str,
) -> bool:
    """
    Detect hallucination in an OE response.

    Cases:
    1. T4-TUMOR OE on healthy subject: asserts any positive pathology finding.
    2. T4-TUMOR OE on tumor subject: asserts pathology but wrong laterality (T5).
    3. T5-LAT OE: asserts a laterality contradicted by GT (for tumor subjects).
    4. T7-ABSTAIN / negative controls: claims findings instead of abstaining.
    """
    if format_ != "OE":
        return False   # Hallucination detection only on open-ended responses

    has_positive_findings = any(f in _POSITIVE_FINDINGS_SET for f in asserted_findings)

    if task == "T4-TUMOR":
        if label_tumor_present == "no" and has_positive_findings:
            return True   # claimed pathology in healthy brain

    if task == "T5-LAT":
        if label_tumor_present == "no" and has_positive_findings:
            return True   # claimed lateralized finding in healthy brain
        if label_tumor_present == "yes" and pred_answer_lat and label_laterality not in ("unknown", "none"):
            # Wrong laterality with confident positive finding
            if pred_answer_lat != label_laterality and pred_answer_lat not in ("none", "UNSURE", "UNPARSEABLE"):
                return True

    if task == "T7-ABSTAIN":
        # Should have abstained; instead asserts positive findings
        if has_positive_findings and pred_answer != "UNSURE":
            return True

    return False


def grade_record(
    parsed_rec: dict[str, Any],
    manifest_row: dict[str, Any],
) -> dict[str, Any]:
    """
    Grade one parsed response record against its manifest ground-truth row.
    Returns the record enriched with grading fields.
    """
    task      = parsed_rec.get("task", "")
    format_   = parsed_rec.get("format", "OE")
    pred      = parsed_rec.get("pred_answer", "UNPARSEABLE")
    conf_raw  = parsed_rec.get("confidence")
    conf      = (conf_raw / 100.0) if conf_raw is not None else None
    abstained = parsed_rec.get("abstained", False)
    findings  = parsed_rec.get("asserted_findings", [])
    is_nc     = bool(manifest_row.get("is_negative_control", 0))

    # Ground-truth label for this task
    gt = _get_gt(task, manifest_row)

    # Gradeable = a usable ground-truth label exists. Items with no/indeterminate GT
    # (e.g. T5-LAT laterality 'unknown' when the seg mask couldn't be read, or T6-VQA
    # which has no per-item GT) must be EXCLUDED from accuracy, not counted as wrong.
    gradeable = str(gt).strip().lower() not in ("", "unknown", "n/a", "nan", "none_gt")

    # ── Correctness ────────────────────────────────────────────────────────────
    # Abstained and unparseable are not counted as wrong; they have their own flags
    if abstained or pred in ("UNPARSEABLE",):
        correct = False
        answered = False
    else:
        correct  = _labels_match(pred, gt)
        answered = True

    # ── Hallucination ─────────────────────────────────────────────────────────
    label_tumor = str(manifest_row.get("label_tumor_present", "no")).lower()
    label_lat   = str(manifest_row.get("label_laterality", "none")).lower()
    pred_lat    = pred if task == "T5-LAT" else None

    hallucination = is_hallucination(
        asserted_findings=findings,
        label_tumor_present=label_tumor,
        pred_answer=pred,
        label_laterality=label_lat,
        pred_answer_lat=pred_lat,
        task=task,
        format_=format_,
    )

    # ── Appropriate abstention ────────────────────────────────────────────────
    should_abstain = is_nc or (task == "T7-ABSTAIN")
    appropriate_abstention = abstained and should_abstain

    # ── Confidently wrong ─────────────────────────────────────────────────────
    conf_pct = conf_raw if conf_raw is not None else 0
    confidently_wrong = (
        conf_pct >= CONFIDENCE_HIGH_THRESHOLD
        and answered
        and not correct
    )

    return {
        **parsed_rec,
        "gt_label":              gt,
        "gradeable":             gradeable,
        "correct":               correct,
        "answered":              answered,
        "hallucination":         hallucination,
        "appropriate_abstention": appropriate_abstention,
        "confidently_wrong":     confidently_wrong,
        "conf_frac":             conf,        # float 0-1 or None
        "is_negative_control":   is_nc,
    }


def _get_gt(task: str, manifest_row: dict) -> str:
    """Extract the ground-truth label string for a given task.

    T4-TUMOR / T5-LAT prefer SLICE-LEVEL labels (`slice_tumor_present` /
    `slice_laterality`, derived from the segmentation mask of the exact slice)
    when present in the manifest — fixing the subject-level labeling bug (R1).
    Falls back to subject-level labels for backward compatibility (old manifests).
    A 'unknown' slice label propagates and is excluded from grading via `gradeable`.
    """
    def _v(key):
        v = manifest_row.get(key)
        s = str(v).strip().lower() if v is not None else ""
        return s if s not in ("", "nan") else ""

    if task == "T4-TUMOR":
        return _v("slice_tumor_present") or _v("label_tumor_present")
    if task == "T5-LAT":
        return _v("slice_laterality") or _v("label_laterality")
    if task == "T1-MOD":
        return _v("label_modality")
    if task == "T2-PLANE":
        return _v("label_plane")
    if task == "T3-ISBRAIN":
        return "no" if manifest_row.get("is_negative_control", 0) else "yes"
    if task == "T7-ABSTAIN":
        return "UNSURE"
    return ""


def _labels_match(pred: str, gt: str) -> bool:
    """Case-insensitive label comparison with lightweight normalization."""
    if not pred or not gt:
        return False
    return pred.strip().lower() == gt.strip().lower()


def grade_batch(
    parsed_records: list[dict[str, Any]],
    manifest: "pd.DataFrame",
) -> list[dict[str, Any]]:
    """
    Grade a list of parsed records. Joins with manifest on image_id to get GT labels.
    """
    import pandas as pd

    mf_dict = manifest.set_index("image_id").to_dict(orient="index")
    graded = []
    for rec in parsed_records:
        image_id = rec.get("image_id", "")
        mrow = mf_dict.get(image_id, {})
        graded.append(grade_record(rec, mrow))
    return graded
