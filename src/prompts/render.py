"""
Render prompts for every (image, task, phrasing, format) combination.

Key behaviours:
  - MC option order is randomized per item (seed = hash of image_id + task + phrasing)
  - The shuffled option list and original indices are recorded for grading
  - OE and MC formats are always both generated
  - Output: list of PromptRow dicts consumed by run_inference.py

Usage:
    from src.prompts.render import render_all
    rows = render_all(prompts_cfg, tasks_cfg, manifest_df, phrasings=['neutral','terse','clinician'])
"""
from __future__ import annotations

import hashlib
import string
from typing import TypedDict

import pandas as pd

from src.utils import load_yaml, get_logger
from src.utils.seeds import shuffle_mc_options

log = get_logger(__name__)

OPTION_LETTERS = list(string.ascii_uppercase)   # A, B, C, ...


class PromptRow(TypedDict):
    prompt_id:       str
    image_id:        str
    task:            str
    phrasing:        str
    format:          str        # MC | OE
    prompt_text:     str
    system_prompt:   str
    mc_options:      list[str]  # empty for OE
    mc_option_order: list[int]  # original indices (empty for OE)
    mc_correct_letter: str      # letter of correct answer (empty for OE)


def _item_seed(image_id: str, task: str, phrasing: str) -> int:
    """Deterministic seed for MC option shuffling, unique per (image, task, phrasing)."""
    h = hashlib.sha256(f"{image_id}|{task}|{phrasing}".encode()).digest()
    return int.from_bytes(h[:4], "big")


def _format_mc_options(options: list[str]) -> str:
    """Format shuffled options as 'A. opt1  B. opt2  ...' single-line string."""
    return "  ".join(f"{OPTION_LETTERS[i]}. {opt}" for i, opt in enumerate(options))


def _correct_letter(shuffled_options: list[str], label: str, aliases: dict[str, list[str]]) -> str:
    """Find the option letter corresponding to the ground-truth label."""
    label_lower = label.lower().strip()
    for i, opt in enumerate(shuffled_options):
        opt_lower = opt.lower().strip()
        if opt_lower == label_lower:
            return OPTION_LETTERS[i]
        # Check aliases
        for canonical, alias_list in aliases.items():
            if opt_lower in [a.lower() for a in alias_list] or opt_lower == canonical.lower():
                if label_lower == canonical.lower() or label_lower in [a.lower() for a in alias_list]:
                    return OPTION_LETTERS[i]
    return "?"   # GT label not found in answer space (should not happen)


def render_row(
    image_row: pd.Series,
    task_id: str,
    task_cfg: dict,
    phrasing: str,
    phrasing_cfg: dict,
    system_prompt: str,
) -> list[PromptRow]:
    """
    Render MC and OE prompt rows for one (image, task, phrasing) combination.
    Returns list of 1-2 PromptRow dicts.
    """
    image_id = image_row["image_id"]
    dataset  = image_row.get("dataset", "")
    label    = _get_label_for_task(image_row, task_id)
    # Coerce to str: YAML parses bare yes/no/on/off as booleans, which would crash
    # the .lower() calls downstream. str() is a no-op for already-string labels.
    aliases  = {
        str(k): [str(v) for v in (vs or [])]
        for k, vs in (task_cfg.get("answer_space_aliases", {}) or {}).items()
    }
    answer_space = [str(a) for a in task_cfg.get("answer_space", [])]

    task_phrasing = phrasing_cfg.get(task_id, {})
    if not task_phrasing:
        return []   # this phrasing has no template for this task

    rows: list[PromptRow] = []

    for fmt in ("mc", "oe"):
        tmpl = task_phrasing.get(fmt, "")
        if not tmpl:
            continue

        # Skip MC for tasks with no closed answer space (e.g. T6-VQA): an MC prompt
        # with no options always parses to UNPARSEABLE, so don't emit it.
        if fmt == "mc" and not answer_space:
            continue

        if fmt == "mc" and answer_space:
            seed = _item_seed(image_id, task_id, phrasing)
            shuffled, original_indices = shuffle_mc_options(answer_space, seed)
            options_str = _format_mc_options(shuffled)
            prompt_text = tmpl.format(options=options_str, question=task_cfg.get("mc_template", ""))
            correct_letter = _correct_letter(shuffled, label, aliases) if label else "?"
            mc_options_list = shuffled
            mc_order = original_indices
        else:
            prompt_text = tmpl.format(
                options="",
                question=task_cfg.get("oe_template", task_cfg.get("name", ""))
            )
            correct_letter = ""
            mc_options_list = []
            mc_order = []

        prompt_id = f"{image_id}__{task_id}__{phrasing}__{fmt}"
        rows.append(PromptRow(
            prompt_id=prompt_id,
            image_id=image_id,
            task=task_id,
            phrasing=phrasing,
            format=fmt.upper(),
            prompt_text=prompt_text.strip(),
            system_prompt=system_prompt.strip(),
            mc_options=mc_options_list,
            mc_option_order=mc_order,
            mc_correct_letter=correct_letter,
        ))

    return rows


def _get_label_for_task(row: pd.Series, task_id: str) -> str:
    """Map task ID to the relevant label column in exam_set.csv."""
    mapping = {
        "T1-MOD":   "label_modality",
        "T2-PLANE": "label_plane",
        "T3-ISBRAIN": None,           # always 'yes' for real brain MRIs
        "T4-TUMOR": "label_tumor_present",
        "T5-LAT":   "label_laterality",
        "T6-VQA":   None,             # dynamic
        "T7-ABSTAIN": None,           # expected: UNSURE
    }
    col = mapping.get(task_id)
    if col is None:
        if task_id == "T3-ISBRAIN":
            return "no" if row.get("is_negative_control", 0) else "yes"
        if task_id == "T7-ABSTAIN":
            return "UNSURE"
        return ""
    return str(row.get(col, "")).strip()


def task_applies(task_cfg: dict, dataset: str, is_negative_control: bool) -> bool:
    """Check whether a task applies to the given dataset row."""
    applies = task_cfg.get("applies_to_datasets", [])
    if not applies:
        return True
    if is_negative_control:
        return "negative_controls" in applies
    return dataset in applies


def render_all(
    prompts_cfg: dict,
    tasks_cfg: dict,
    manifest: pd.DataFrame,
    phrasings: list[str] | None = None,
) -> list[PromptRow]:
    """
    Generate the full prompt matrix: manifest rows × tasks × phrasings × {MC, OE}.
    Returns flat list of PromptRow dicts.
    """
    phrasings = phrasings or ["neutral", "terse", "clinician"]
    system_prompt = prompts_cfg.get("system_prompt", "")

    all_rows: list[PromptRow] = []
    skipped = 0

    for _, img_row in manifest.iterrows():
        dataset = img_row.get("dataset", "")
        is_nc   = bool(img_row.get("is_negative_control", 0))

        for task_id, task_cfg in tasks_cfg.get("tasks", {}).items():
            if not task_applies(task_cfg, dataset, is_nc):
                continue
            for phrasing in phrasings:
                phrasing_cfg = prompts_cfg.get("phrasings", {}).get(phrasing, {})
                prompt_rows = render_row(
                    image_row=img_row,
                    task_id=task_id,
                    task_cfg=task_cfg,
                    phrasing=phrasing,
                    phrasing_cfg=phrasing_cfg,
                    system_prompt=system_prompt,
                )
                all_rows.extend(prompt_rows)

    log.info(f"Rendered {len(all_rows)} prompt rows ({skipped} skipped)")
    return all_rows


def render_few_shot_prefix(
    support_rows: list[dict],
    task_id: str,
    phrasing: str,
    prompts_cfg: dict,
    tasks_cfg: dict,
    n_shots: int = 2,
) -> str:
    """
    Build few-shot ICL prefix (ablation A5).
    Returns a text block with n_shots example Q/A pairs to prepend to the prompt.
    """
    support_rows = support_rows[:n_shots]
    parts = ["Here are example cases:\n"]
    for i, ex in enumerate(support_rows):
        parts.append(f"Example {i+1}:")
        parts.append(f"  Question: {ex.get('prompt_text', '')[:200]}")
        parts.append(f"  Answer: {ex.get('label', '')} (Confidence: {ex.get('example_confidence', 90)})")
    parts.append("\nNow answer the following:\n")
    return "\n".join(parts)


if __name__ == "__main__":
    import sys
    from src.utils import load_yaml
    prompts_cfg = load_yaml("config/prompts.yaml")
    tasks_cfg   = load_yaml("config/tasks.yaml")
    manifest    = pd.read_csv("data/exam_set.csv")
    rows = render_all(prompts_cfg, tasks_cfg, manifest)
    print(f"Total prompt rows: {len(rows)}")
    if rows:
        print("\nSample row:")
        r = rows[0]
        print(f"  prompt_id: {r['prompt_id']}")
        print(f"  format:    {r['format']}")
        print(f"  task:      {r['task']}")
        print(f"  text:\n{r['prompt_text'][:300]}")
