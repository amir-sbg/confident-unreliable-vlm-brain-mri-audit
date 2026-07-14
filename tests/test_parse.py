"""
Parser unit tests — must pass before scaling to the full exam set.
Tests ~30 synthetic response scenarios including:
  - clean MC/OE answers
  - hedging/uncertainty language
  - multiple answer mentions
  - missing confidence
  - unit-bearing hallucinations (tumor/mass in healthy subjects)
  - UNSURE / refusal detection
  - edge cases (empty, garbled, non-English artifacts)
"""
from __future__ import annotations

import pytest
from src.scoring.parse import (
    extract_confidence,
    detect_abstention,
    extract_asserted_findings,
    normalize_to_answer_space,
    extract_mc_answer,
    parse_response,
)

# ── extract_confidence ─────────────────────────────────────────────────────────

def test_confidence_standard():
    assert extract_confidence("Confidence: 85") == 85

def test_confidence_percent():
    assert extract_confidence("My confidence is 72%") == 72

def test_confidence_with_slash():
    assert extract_confidence("Confidence: 90/100") == 90

def test_confidence_certainty():
    assert extract_confidence("Certainty: 45") == 45

def test_confidence_at_end():
    assert extract_confidence("This is T1. Confidence: 100") == 100

def test_confidence_missing():
    assert extract_confidence("This is a brain MRI.") is None

def test_confidence_out_of_range_ignored():
    # 150 is out of range, should not be returned
    result = extract_confidence("Confidence: 150")
    assert result is None

def test_confidence_zero():
    assert extract_confidence("Confidence: 0") == 0

# ── detect_abstention ──────────────────────────────────────────────────────────

def test_abstain_unsure_token():
    assert detect_abstention("UNSURE") is True

def test_abstain_unsure_lowercase():
    assert detect_abstention("unsure") is True

def test_abstain_refusal():
    assert detect_abstention("I cannot determine the sequence from this image.") is True

def test_abstain_not_suitable():
    assert detect_abstention("This image is not suitable for clinical evaluation.") is True

def test_abstain_unable():
    assert detect_abstention("I am unable to assess this image.") is True

def test_abstain_not_brain():
    assert detect_abstention("This is not a brain MRI image.") is True

def test_no_abstain_confident():
    assert detect_abstention("This is a T1-weighted image. Confidence: 90") is False

def test_no_abstain_hedged_but_not_refusing():
    # "I believe" is not a refusal
    assert detect_abstention("I believe this is an axial T2 image.") is False

# ── extract_asserted_findings ──────────────────────────────────────────────────

def test_findings_tumor_detected():
    text = "I can see a large tumor in the right hemisphere."
    findings = extract_asserted_findings(text)
    assert "tumor" in findings

def test_findings_mass_effect():
    text = "There is significant mass effect with midline shift."
    findings = extract_asserted_findings(text)
    assert "mass effect" in findings or "midline shift" in findings

def test_findings_negated_not_included():
    text = "There is no tumor or mass evident in this image."
    findings = extract_asserted_findings(text)
    # Should not include tumor/mass since they are negated
    assert "tumor" not in findings
    assert "mass" not in findings

def test_findings_edema():
    text = "Perilesional edema is present surrounding the lesion."
    findings = extract_asserted_findings(text)
    assert "edema" in findings or "perilesional edema" in findings

def test_findings_normal_brain():
    text = "The brain appears normal. No abnormalities detected."
    findings = extract_asserted_findings(text)
    assert len(findings) == 0

def test_findings_multiple():
    text = "A large glioma is present with surrounding edema and midline shift."
    findings = extract_asserted_findings(text)
    assert len(findings) >= 2

# ── normalize_to_answer_space ──────────────────────────────────────────────────

def test_normalize_exact():
    assert normalize_to_answer_space("T1", ["T1", "T2", "FLAIR"]) == "T1"

def test_normalize_case_insensitive():
    assert normalize_to_answer_space("t2", ["T1", "T2", "FLAIR"]) == "T2"

def test_normalize_alias():
    assert normalize_to_answer_space("flair sequence", ["T1", "T2", "FLAIR"]) == "FLAIR"

def test_normalize_plane_axial():
    assert normalize_to_answer_space("transverse", ["axial", "sagittal", "coronal"]) == "axial"

def test_normalize_yes_aliases():
    ans = normalize_to_answer_space("present", ["yes", "no"])
    assert ans == "yes"

def test_normalize_no_match():
    assert normalize_to_answer_space("elephant", ["T1", "T2", "FLAIR"]) is None

def test_normalize_none_answer():
    assert normalize_to_answer_space("none", ["left", "right", "bilateral", "none"]) == "none"

def test_normalize_with_letter_prefix():
    # "A. T1" -> "T1"
    assert normalize_to_answer_space("A. T1", ["T1", "T2", "FLAIR"]) == "T1"

# ── extract_mc_answer ──────────────────────────────────────────────────────────

def test_mc_letter_A():
    assert extract_mc_answer("A. T1 weighted", ["T1", "T2", "FLAIR"]) == "T1"

def test_mc_letter_B():
    assert extract_mc_answer("B", ["axial", "sagittal", "coronal"]) == "sagittal"

def test_mc_answer_is_pattern():
    assert extract_mc_answer("Answer is B", ["left", "right", "bilateral", "none"]) == "right"

def test_mc_answer_colon():
    assert extract_mc_answer("Answer: C", ["yes", "no", "UNSURE"]) == "UNSURE"

def test_mc_fallback_to_label():
    # No letter found, but label text present
    result = extract_mc_answer("The image is in the axial plane.", ["axial", "sagittal", "coronal"])
    assert result == "axial"

def test_mc_empty_response():
    assert extract_mc_answer("", ["yes", "no"]) is None

# ── parse_response (integration) ──────────────────────────────────────────────

def test_parse_mc_clean():
    result = parse_response(
        "B. T2\nConfidence: 88",
        task="T1-MOD",
        format_="MC",
        answer_space=["T1", "T2", "FLAIR"],
        mc_options=["T1", "T2", "FLAIR"],
    )
    assert result["pred_answer"] == "T2"
    assert result["confidence"] == 88
    assert result["abstained"] is False
    assert result["unparseable"] is False

def test_parse_oe_tumor():
    result = parse_response(
        "I can identify a large mass lesion in the right hemisphere with surrounding edema.\nConfidence: 92",
        task="T4-TUMOR",
        format_="OE",
        answer_space=["yes", "no"],
    )
    assert result["confidence"] == 92
    assert len(result["asserted_findings"]) > 0

def test_parse_abstain():
    result = parse_response(
        "UNSURE - this image appears corrupted.",
        task="T7-ABSTAIN",
        format_="OE",
        answer_space=["UNSURE"],
    )
    assert result["abstained"] is True
    assert result["pred_answer"] == "UNSURE"

def test_parse_missing_confidence():
    result = parse_response(
        "This is a coronal slice.",
        task="T2-PLANE",
        format_="OE",
        answer_space=["axial", "sagittal", "coronal"],
    )
    assert result["pred_answer"] == "coronal"
    assert result["confidence"] is None
    assert result["unparseable"] is False

def test_parse_unparseable():
    result = parse_response(
        "I'm not sure what this is. Maybe something medical. Confidence: 30",
        task="T1-MOD",
        format_="OE",
        answer_space=["T1", "T2", "FLAIR", "PD", "MRA", "DWI"],
    )
    assert result["unparseable"] is True

def test_parse_multiple_mentions_first_wins():
    # Letter "B" maps to index 1 = T2 in the shuffled options list
    result = parse_response(
        "B. T2-weighted MRI.\nConfidence: 75",
        task="T1-MOD",
        format_="MC",
        answer_space=["T1", "T2", "FLAIR"],
        mc_options=["T1", "T2", "FLAIR"],
    )
    assert result["pred_answer"] == "T2"

def test_parse_hallucinated_tumor_in_healthy_oe():
    """OE response asserting tumor on a healthy subject (caught at grading, not parsing)."""
    result = parse_response(
        "I see a tumor mass with edema in the left hemisphere. Confidence: 95",
        task="T4-TUMOR",
        format_="OE",
        answer_space=["yes", "no"],
    )
    # Parsing should extract findings even if they are hallucinated
    assert "tumor" in result["asserted_findings"] or "edema" in result["asserted_findings"]
    assert result["abstained"] is False

def test_parse_empty():
    result = parse_response("", task="T1-MOD", format_="MC",
                            answer_space=["T1", "T2"], mc_options=["T1", "T2"])
    assert result["unparseable"] is True or result["abstained"] is True

def test_parse_garbled():
    result = parse_response(
        "asdf lkjh qwerty 123456 @#$%",
        task="T2-PLANE",
        format_="OE",
        answer_space=["axial", "sagittal", "coronal"],
    )
    assert result["unparseable"] is True

def test_parse_cannot_determine():
    result = parse_response(
        "I cannot determine the imaging sequence from this image.",
        task="T1-MOD",
        format_="OE",
        answer_space=["T1", "T2", "FLAIR"],
    )
    assert result["abstained"] is True


# ── Run directly ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
