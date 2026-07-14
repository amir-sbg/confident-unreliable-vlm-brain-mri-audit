"""
Parse raw VLM responses into structured fields:
  - pred_answer: normalized to task answer space
  - confidence: int 0-100
  - asserted_findings: lexicon-matched phrases (for hallucination detection)
  - abstained: bool (UNSURE token or refusal detected)
  - unparseable: bool

Used in both aggregate.py and tests/test_parse.py.
"""
from __future__ import annotations

import re
from typing import Any

# ── Answer-space normalization ────────────────────────────────────────────────

ANSWER_ALIASES: dict[str, list[str]] = {
    # Modality
    "T1":    ["t1", "t1w", "t1-weighted", "t1weighted", "t1 weighted",
               "t1 image", "spin echo t1"],
    "T1ce":  ["t1ce", "t1c", "t1-ce", "t1 post-contrast", "post-contrast",
               "post contrast", "contrast-enhanced t1", "t1 with contrast",
               "t1 gad", "gadolinium-enhanced"],
    "T2":    ["t2", "t2w", "t2-weighted", "t2weighted", "t2 weighted"],
    "FLAIR": ["flair", "fluid attenuated", "fluid-attenuated inversion recovery",
               "flair sequence"],
    "PD":    ["pd", "proton density", "proton-density", "pd weighted"],
    "MRA":   ["mra", "magnetic resonance angiography", "angiography", "angio"],
    "DWI":   ["dwi", "diffusion", "diffusion weighted", "diffusion-weighted",
               "adc", "apparent diffusion"],
    # Plane
    "axial":    ["axial", "transverse", "transaxial", "horizontal", "cross-sectional"],
    "sagittal": ["sagittal", "lateral", "midsagittal", "mid-sagittal"],
    "coronal":  ["coronal", "frontal", "anterior"],
    # Boolean
    "yes": ["yes", "present", "detected", "visible", "there is", "shows",
             "demonstrates", "abnormal", "confirms", "consistent with", "positive"],
    "no":  ["no", "absent", "not detected", "normal", "not present", "none",
             "negative", "unremarkable", "without", "no evidence"],
    # Laterality
    "left":      ["left", "left hemisphere", "left-sided", "left side", "lh"],
    "right":     ["right", "right hemisphere", "right-sided", "right side", "rh"],
    "bilateral": ["bilateral", "both hemispheres", "both sides", "bihemispheric", "diffuse"],
    "none":      ["none", "n/a", "not applicable", "no abnormality", "no lesion"],
}

# Reverse lookup: alias -> canonical
_ALIAS_TO_CANONICAL: dict[str, str] = {}
for canonical, aliases in ANSWER_ALIASES.items():
    for alias in aliases:
        _ALIAS_TO_CANONICAL[alias.lower()] = canonical
    _ALIAS_TO_CANONICAL[canonical.lower()] = canonical

# ── Confidence extraction ─────────────────────────────────────────────────────

_CONFIDENCE_PATTERNS = [
    re.compile(r"confidence[:\s]+(\d{1,3})\s*(?:/\s*100|%)?", re.IGNORECASE),
    re.compile(r"(\d{1,3})\s*%?\s*confidence", re.IGNORECASE),
    re.compile(r"my confidence.*?(\d{1,3})", re.IGNORECASE),
    re.compile(r"certainty[:\s]+(\d{1,3})", re.IGNORECASE),
]

_ABSTAIN_PATTERNS = [
    re.compile(r"\bUNSURE\b", re.IGNORECASE),
    re.compile(r"\bI cannot\b.*\b(determine|identify|assess)\b", re.IGNORECASE),
    re.compile(r"\b(not\s+a\s+brain|not\s+suitable|non-diagnostic|uninterpretable"
               r"|cannot\s+be\s+evaluated|too\s+poor|insufficient\s+quality)\b", re.IGNORECASE),
    re.compile(r"\b(unable to|not able to)\b.*\b(assess|determine|identify|interpret)\b",
               re.IGNORECASE),
    re.compile(r"\bI\s+don[\'']t\s+have\s+enough\b", re.IGNORECASE),
    re.compile(r"\bI\s+cannot\s+provide\b", re.IGNORECASE),
]

# ── Hallucination lexicon ─────────────────────────────────────────────────────

POSITIVE_FINDING_PHRASES = [
    "tumor", "mass", "lesion", "neoplasm", "glioma", "glioblastoma",
    "meningioma", "metastasis", "metastases", "hemorrhage", "bleed",
    "hematoma", "infarct", "infarction", "stroke", "edema", "enhancement",
    "ring-enhancing", "mass effect", "midline shift", "herniation",
    "atrophy", "leukoencephalopathy", "demyelination", "abscess",
    "hydrocephalus", "subdural", "epidural", "subarachnoid",
    "signal abnormality", "signal intensity", "hypointense lesion",
    "hyperintense lesion", "abnormal signal", "contrast enhancement",
    "heterogeneous signal", "restricted diffusion", "T2 hyperintensity",
    "white matter changes", "perilesional edema", "vasogenic edema",
]

_POSITIVE_FINDING_RE = re.compile(
    r"\b(" + "|".join(re.escape(p) for p in POSITIVE_FINDING_PHRASES) + r")\b",
    re.IGNORECASE,
)

NEGATIVE_PHRASES = [
    "normal", "unremarkable", "no abnormality", "no evidence of",
    "no lesion", "no mass", "no tumor", "no pathology",
    "no significant", "within normal limits",
]
_NEGATIVE_RE = re.compile(
    r"\b(" + "|".join(re.escape(p) for p in NEGATIVE_PHRASES) + r")\b",
    re.IGNORECASE,
)


def extract_confidence(text: str) -> int | None:
    """Extract confidence integer (0-100) from response text. Returns None if not found."""
    for pat in _CONFIDENCE_PATTERNS:
        m = pat.search(text)
        if m:
            val = int(m.group(1))
            if 0 <= val <= 100:
                return val
    return None


def detect_abstention(text: str) -> bool:
    """Return True if the response contains an abstention / UNSURE signal."""
    for pat in _ABSTAIN_PATTERNS:
        if pat.search(text):
            return True
    return False


def extract_asserted_findings(text: str) -> list[str]:
    """
    Return list of positive finding phrases found in OE response text.
    Skips phrases preceded by a negation (crude but effective).
    """
    matches = _POSITIVE_FINDING_RE.finditer(text)
    findings = []
    for m in matches:
        start = max(0, m.start() - 40)
        context = text[start:m.start()].lower()
        # Skip if negation appears within ~40 chars before the finding
        if re.search(r"\b(no|not|without|absent|deny|denies|negative for)\b", context):
            continue
        findings.append(m.group(0).lower())
    return list(dict.fromkeys(findings))   # deduplicate preserving order


def normalize_to_answer_space(text: str, answer_space: list[str]) -> str | None:
    """
    Normalize raw text to a canonical answer-space label.
    Strategy:
      1. Exact match (case-insensitive)
      2. Alias match (via ANSWER_ALIASES)
      3. Substring search in answer space
    Returns None if no match found.
    """
    if not text or not answer_space:
        return None

    clean = text.strip().lower()
    # Remove leading letter prefix (e.g. "A. T1" -> "T1")
    clean = re.sub(r"^[A-Z]\.\s*", "", clean.strip())
    clean = clean.strip().strip("*").strip()

    answer_lower = {a.lower(): a for a in answer_space}

    # Exact match
    if clean in answer_lower:
        return answer_lower[clean]

    # Alias match
    canonical = _ALIAS_TO_CANONICAL.get(clean)
    if canonical and canonical in answer_space:
        return canonical

    # Try longer alias phrases
    for alias, canon in sorted(_ALIAS_TO_CANONICAL.items(), key=lambda x: -len(x[0])):
        if alias in clean and canon in answer_space:
            return canon

    # Substring: answer space label appears in response text
    for label in sorted(answer_space, key=len, reverse=True):
        if label.lower() in clean:
            return label

    return None


def extract_mc_answer(text: str, mc_options: list[str]) -> str | None:
    """
    For MC responses: try to extract the chosen option letter or label.
    Returns the canonical label string, or None.
    """
    if not text or not mc_options:
        return None

    clean = text.strip()

    # Pattern: "Answer: B" or "My answer is B" — check first (higher precision)
    m = re.search(r"(?:answer\s*(?:is|:)?\s*)([A-Z])\b", clean, re.IGNORECASE)
    if m:
        letter = m.group(1).upper()
        idx = ord(letter) - ord("A")
        if 0 <= idx < len(mc_options):
            return mc_options[idx]

    # Pattern: starts with option letter followed by separator or end-of-string
    # e.g. "B", "B.", "B. sagittal", "B) sagittal"  — NOT "Bilateral" or "Brain"
    m = re.match(r"^([A-Z])(?:[.\):,]|\s|$)", clean)
    if m:
        letter = m.group(1)
        idx = ord(letter) - ord("A")
        if 0 <= idx < len(mc_options):
            return mc_options[idx]

    # Fallback: normalize text against option labels
    return normalize_to_answer_space(clean, mc_options)


def parse_response(
    response_text: str,
    task: str,
    format_: str,          # "MC" or "OE"
    answer_space: list[str],
    mc_options: list[str] | None = None,
) -> dict[str, Any]:
    """
    Parse one raw VLM response into structured fields.

    Returns dict with:
      pred_answer, confidence, asserted_findings, abstained, unparseable
    """
    text = response_text or ""

    abstained = detect_abstention(text)
    confidence = extract_confidence(text)

    if abstained:
        return {
            "pred_answer": "UNSURE",
            "confidence": confidence,
            "asserted_findings": [],
            "abstained": True,
            "unparseable": False,
        }

    # Extract answer
    if format_ == "MC" and mc_options:
        pred = extract_mc_answer(text, mc_options)
    else:
        pred = normalize_to_answer_space(text, answer_space)

    asserted = extract_asserted_findings(text) if format_ == "OE" else []

    return {
        "pred_answer": pred if pred else "UNPARSEABLE",
        "confidence": confidence,
        "asserted_findings": asserted,
        "abstained": False,
        "unparseable": pred is None,
    }


def parse_batch(records: list[dict], tasks_cfg: dict) -> list[dict]:
    """
    Parse a batch of raw response records. Each record must have:
      model, image_id, prompt_id, task, format, response, mc_options, ...
    Returns list of records enriched with parsed fields.
    """
    out = []
    for rec in records:
        task = rec.get("task", "")
        fmt  = rec.get("format", "OE")
        task_def = tasks_cfg.get("tasks", {}).get(task, {})
        # Coerce to str (YAML yes/no parse as booleans); no-op for real strings.
        answer_space = [str(a) for a in task_def.get("answer_space", [])]
        mc_options   = [str(o) for o in rec.get("mc_options", [])]

        parsed = parse_response(
            response_text=rec.get("response", ""),
            task=task,
            format_=fmt,
            answer_space=answer_space,
            mc_options=mc_options,
        )
        out.append({**rec, **parsed})
    return out
