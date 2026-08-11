# Confident but Unreliable: A Behavioral Safety Audit of Vision-Language Models on Brain MRI

<p align="center">
  <img src="docs/logos/trends.png" alt="TReNDS Center" height="42">
  &nbsp;&nbsp;&nbsp;
  <img src="docs/logos/georgia-state.jpg" alt="Georgia State University" height="42">
  &nbsp;&nbsp;&nbsp;
  <img src="docs/logos/georgia-tech.png" alt="Georgia Tech" height="42">
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2608.02790"><img src="https://img.shields.io/badge/arXiv-2608.02790-b31b1b.svg" alt="arXiv"></a>
  <a href="https://aisummit.acm.org/"><img src="https://img.shields.io/badge/ACM%20AI-2026-0055A4.svg" alt="ACM AI 2026"></a>
  <img src="https://img.shields.io/badge/Python-3.11-3776AB.svg" alt="Python 3.11">
  <img src="https://img.shields.io/badge/status-research%20code-lightgrey.svg" alt="Research code">
</p>

Official research code for the paper:

> **Confident but Unreliable: A Behavioral Safety Audit of Vision-Language Models on Brain MRI**<br>
> Amir Sabbaghziarani, Mohammadsajad Abavisani, Sergey Plis<br>
> Accepted to **ACM AI 2026**. arXiv: [2608.02790](https://arxiv.org/abs/2608.02790)

This repository contains the research code for an inference-only safety audit of vision-language models (VLMs) on brain MRI. The benchmark asks whether models can produce correct answers, calibrated confidence, and appropriate abstentions on controlled MRI-derived images.

> This is research evaluation code. It is **not** a clinical diagnostic system and should not be used for patient-care decisions.

## Summary

The audit evaluates six instruction-tuned VLMs on **4,102 images**:

- **4,032 MRI slices** from **250 subjects**
- **70 non-brain/noise controls**
- Labels derived from public metadata, slice geometry, and released segmentation masks
- Multiple-choice and open-ended prompts with explicit confidence elicitation

The main question is not only whether a model is correct. It is whether the model knows when it is wrong.

## Key Findings

- Expected Calibration Error (ECE) ranges from **0.27 to 0.40**.
- Mean confidence on wrong answers ranges from **0.82 to 0.97**.
- **33-46%** of answered items are high-confidence errors.
- The most accurate audited model is also the most confident on its mistakes.
- Medical adaptation improves tumor-presence detection in one family comparison, but does not improve confidence reliability.
- Open-ended hallucination and abstention behavior vary separately from multiple-choice accuracy.

<p align="center">
  <img src="docs/accuracy-vs-error-confidence.png" alt="Overall accuracy versus mean confidence on wrong answers" width="560">
</p>

## Benchmark

- **Data**: BraTS 2024, IXI, and non-brain/noise controls. OASIS is supported as an optional extension.
- **Models**: `InternVL2.5-8B`, `Qwen2.5-VL-3B`, `Qwen2.5-VL-7B`, `Gemma-3-4B`, `Gemma-4-12B`, and `MedGemma-4B`.
- **Tasks**: MRI sequence, imaging plane, brain-vs-non-brain recognition, tumor presence, lesion laterality, and abstention/OOD behavior.
- **Metrics**: coverage, accuracy, macro-F1, ECE, Brier score, confidence on wrong answers, confidently-wrong rate, hallucination, and abstention.

## Headline Results

Neutral multiple-choice prompt results from the paper:

| Model | Coverage | Accuracy | ECE (lower better) | Brier (lower better) | Confidence on wrong (lower better) | Confidently wrong rate (lower better) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| InternVL2.5-8B | 1.000 | 0.567 | 0.361 | 0.371 | 0.917 | 0.426 |
| Qwen2.5-VL-3B | 1.000 | 0.514 | 0.361 | 0.355 | 0.844 | 0.357 |
| Qwen2.5-VL-7B | 1.000 | 0.563 | 0.272 | 0.326 | 0.819 | 0.360 |
| Gemma-3-4B | 1.000 | 0.531 | 0.404 | 0.405 | 0.922 | 0.462 |
| Gemma-4-12B | 0.996 | 0.670 | 0.307 | 0.311 | 0.968 | 0.330 |
| MedGemma-4B | 1.000 | 0.571 | 0.381 | 0.389 | 0.949 | 0.428 |

Task-resolved patterns:

- Brain-vs-non-brain recognition is easiest for most general models.
- Laterality remains near chance across all audited models.
- Tumor-presence performance is mixed and strongly model-dependent.
- MedGemma improves over Gemma-3 on tumor presence but not confidence reliability.
- Gemma-4-12B is most accurate overall but has the highest confidence on wrong answers.

## Repository

- `config/`: datasets, models, prompts, and task definitions
- `src/data/`: slice extraction and manifest construction
- `src/inference/`: local and API VLM inference runners
- `src/scoring/`: answer parsing, grading, and safety metrics
- `src/analysis/`: aggregation and paper tables
- `slurm/`: cluster launchers for larger sweeps
- `tests/`: parser tests

Large datasets, model weights, raw responses, generated figures, and local logs are ignored by git.

## Quick Start

```bash
git clone https://github.com/amir-sbg/confident-unreliable-vlm-brain-mri-audit.git
cd confident-unreliable-vlm-brain-mri-audit

conda env create -f environment.yml
conda activate vlmaudit
export PYTHONPATH="$PWD"
```

Run the parser tests:

```bash
python -m pytest tests/test_parse.py -q
```

## Minimal Pipeline

This repository does not redistribute medical datasets. Place local datasets under `data/`, then build slices and the exam manifest:

```bash
python -m src.data.slice_extract \
  --config config/datasets.yaml \
  --dataset brats ixi negative_controls \
  --output-dir data/slices \
  --provenance-out data/slice_provenance.jsonl
```

Build the exam manifest:

```bash
python -m src.data.build_manifest \
  --provenance data/slice_provenance.jsonl \
  --config config/datasets.yaml \
  --output data/exam_set.csv
```

Run a model:

```bash
python -m src.inference.run_inference \
  --models config/models.yaml \
  --tasks config/tasks.yaml \
  --prompts config/prompts.yaml \
  --manifest data/exam_set.csv \
  --output-dir raw_responses \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --phrasing neutral \
  --format MC OE \
  --batch-size 8
```

Aggregate and generate paper tables:

```bash
python -m src.analysis.aggregate \
  --responses-dir raw_responses \
  --manifest data/exam_set.csv \
  --tasks config/tasks.yaml \
  --models config/models.yaml \
  --output-dir results

python -m src.analysis.paper_aggregate \
  --responses-dir raw_responses \
  --manifest data/exam_set.csv \
  --tasks config/tasks.yaml \
  --output-dir results/paper
```

The inference runner is resumable: existing `(model, image_id, prompt_id)` records are skipped. For larger sweeps, see `slurm/`.

## Reproducibility Notes

- Prompt option order is randomized per item and logged.
- Inference uses deterministic decoding for the paper audit.
- Raw model outputs are stored as JSONL for auditability.

## Scope and Responsible Use

This benchmark measures behavioral reliability of VLM outputs on controlled brain-MRI-derived images. It does not validate any model for diagnosis, triage, treatment planning, or autonomous clinical use.

## Citation

If you use this repository or build on the audit protocol, please cite:

```bibtex
@misc{sabbaghziarani2026confident,
  title         = {Confident but Unreliable: A Behavioral Safety Audit of Vision-Language Models on Brain MRI},
  author        = {Sabbaghziarani, Amir and Abavisani, Mohammadsajad and Plis, Sergey},
  year          = {2026},
  eprint        = {2608.02790},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2608.02790},
  note          = {Accepted to ACM AI Leadership Summit 2026}
}
```

## Acknowledgments

This work was developed in the TReNDS research environment with collaborators from Georgia State University and Georgia Tech. The audit builds on publicly available neuroimaging resources, including BraTS and IXI, and uses released metadata and segmentation masks to avoid new manual labeling.
