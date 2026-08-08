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
> Accepted to **ACM AI Leadership Summit 2026**. arXiv: [2608.02790](https://arxiv.org/abs/2608.02790)

This repository implements an inference-only benchmark for auditing whether vision-language models (VLMs) know when they are wrong on brain MRI. The central finding is simple and safety-critical: models can answer nearly every question, sound confident, and still be poorly calibrated. Accuracy alone is not enough.

> This is a research evaluation framework. It is **not** a clinical diagnostic system and should not be used for patient-care decisions.

## Overview

Medical VLMs are often evaluated by task accuracy, but downstream users also need to know whether a model's stated confidence is reliable, whether it hallucinates unsupported pathology, and whether it abstains on invalid inputs. This project uses brain MRI as a controlled high-stakes testbed because many labels can be derived automatically from public metadata and expert segmentation masks.

The audit evaluates six instruction-tuned VLMs on **4,102 images**:

- **4,032 MRI slices** from **250 subjects**
- **70 non-brain/noise controls**
- Slice-level labels derived without new manual annotation
- Multiple-choice and open-ended prompt formats
- Deterministic inference with explicit confidence elicitation

<p align="center">
  <img src="docs/pipeline-overview.png" alt="Brain MRI VLM audit pipeline" width="900">
</p>

## Key Findings

The paper shows that confidence and correctness are strongly decoupled:

- Answer coverage is near-complete, so most models rarely decline to answer.
- Expected Calibration Error (ECE) ranges from **0.27 to 0.40**.
- Mean confidence on wrong answers ranges from **0.82 to 0.97**.
- **33-46%** of answered items are high-confidence errors.
- The most accurate audited model is also the most confident on its mistakes.
- Medical adaptation improves tumor-presence detection in one family comparison, but does not improve confidence reliability.
- Open-ended hallucination and abstention behavior vary separately from multiple-choice accuracy.

<p align="center">
  <img src="docs/accuracy-vs-error-confidence.png" alt="Overall accuracy versus mean confidence on wrong answers" width="620">
</p>

## Audit Design

### Data Sources

| Source | Role in the audit | Paper setting |
| --- | --- | --- |
| BraTS 2024 | Tumor-bearing MRI volumes with expert segmentation masks | 1,350 slices from 150 subjects |
| IXI | Healthy volunteer MRI controls | 2,682 slices from 100 subjects |
| Negative controls | Non-brain and synthetic/noise images for abstention/OOD behavior | 70 controls |
| OASIS | Optional supported extension in this codebase | Not required for the headline paper results |

Slices are extracted at controlled fractional depths, reoriented consistently, intensity-normalized, resized to 512 x 512 PNGs, and tracked with provenance. Tumor and laterality labels are slice-level, not subject-level: a slice is tumor-positive only when the segmentation mask is visible in that extracted image.

### Tasks

| ID | Task | Answer space | Why it matters |
| --- | --- | --- | --- |
| `T1-MOD` | MRI sequence identification | T1, T1ce, T2, FLAIR, PD, MRA, DWI | Tests modality/contrast recognition |
| `T2-PLANE` | Imaging plane identification | axial, sagittal, coronal | Tests spatial/anatomical understanding |
| `T3-ISBRAIN` | Brain MRI verification | yes/no | Tests basic OOD and image-type recognition |
| `T4-TUMOR` | Tumor or gross abnormality detection | yes/no | Tests visual pathology detection |
| `T5-LAT` | Lesion laterality | left, right, bilateral, none | Tests localization from a single slice |
| `T6-VQA` | Mixed clinical VQA | dynamic | Optional exploratory extension |
| `T7-ABSTAIN` | Abstention/OOD detection | `UNSURE` | Tests whether models decline unsuitable inputs |

Each task can be rendered as multiple-choice (`MC`) or open-ended (`OE`), with neutral, terse, and clinician-framed prompt phrasings.

### Models

The paper audits six instruction-tuned VLMs:

- `InternVL2.5-8B`
- `Qwen2.5-VL-3B`
- `Qwen2.5-VL-7B`
- `Gemma-3-4B`
- `Gemma-4-12B`
- `MedGemma-4B`

The model registry in `config/models.yaml` also includes optional larger, API-based, and ablation-only models. Optional models are skipped unless explicitly requested.

### Metrics

The benchmark reports more than raw accuracy:

- **Coverage**: fraction of gradeable prompts that receive a parseable answer
- **Accuracy and balanced accuracy**: task competence under class imbalance
- **Macro-F1**: class-balanced summary for multi-class tasks
- **ECE and Brier score**: verbalized-confidence calibration
- **Mean confidence on wrong answers**: how strongly the model stands behind mistakes
- **Confidently-wrong rate**: fraction of answered items that are wrong with confidence >= 0.80
- **Open-ended hallucination rate**: unsupported pathology assertions
- **Abstention appropriateness**: willingness to respond `UNSURE` on invalid/non-brain controls
- **Subject-clustered bootstrap intervals**: uncertainty estimates that respect within-subject correlation

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

## Repository Structure

```text
.
|-- config/
|   |-- datasets.yaml      # data roots, sampling, slice extraction, label rules
|   |-- models.yaml        # model registry and backend settings
|   |-- prompts.yaml       # prompt phrasings and confidence elicitation
|   `-- tasks.yaml         # task taxonomy and answer spaces
|-- docs/
|   |-- pipeline-overview.png
|   `-- accuracy-vs-error-confidence.png
|-- slurm/                 # cluster launchers for larger sweeps
|-- src/
|   |-- analysis/          # aggregation, paper tables, figures
|   |-- data/              # download checks, slice extraction, manifest building
|   |-- inference/         # vLLM, Hugging Face, and API runners
|   |-- prompts/           # prompt rendering
|   |-- scoring/           # response parsing, grading, metrics
|   `-- utils/
|-- tests/
|-- environment.yml
`-- requirements.txt
```

Large datasets, model weights, raw responses, generated figures, and local logs are intentionally ignored by git.

## Installation

```bash
git clone https://github.com/amir-sbg/confident-unreliable-vlm-brain-mri-audit.git
cd confident-unreliable-vlm-brain-mri-audit

conda env create -f environment.yml
conda activate vlmaudit
export PYTHONPATH="$PWD"
```

For CUDA environments, install the appropriate PyTorch build for your driver before running large local VLMs. API models require their provider keys in the environment.

Run the parser tests:

```bash
python -m pytest tests/test_parse.py -q
```

## Data Preparation

This repository does not redistribute medical datasets. Place or download datasets under `data/`.

Check local dataset availability:

```bash
python -m src.data.download \
  --dataset brats ixi oasis \
  --output-dir data \
  --check-only
```

Download IXI where available:

```bash
python -m src.data.download \
  --dataset ixi \
  --output-dir data
```

Build the slice set and provenance file:

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

For older manifests, recompute slice-level tumor/laterality labels in place:

```bash
python -m src.data.augment_slice_labels \
  --manifest data/exam_set.csv \
  --provenance data/slice_provenance.jsonl \
  --area-threshold 25
```

## Running Inference

List configured models:

```bash
python -m src.inference.run_inference \
  --models config/models.yaml \
  --list-models
```

Run one model on neutral MC and OE prompts:

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

The runner is resumable: existing `(model, image_id, prompt_id)` records are skipped. For cluster execution, see the launchers in `slurm/`.

## Scoring and Paper Tables

Aggregate raw responses into parsed, graded, and diagnostic CSV files:

```bash
python -m src.analysis.aggregate \
  --responses-dir raw_responses \
  --manifest data/exam_set.csv \
  --tasks config/tasks.yaml \
  --models config/models.yaml \
  --output-dir results
```

Generate the paper-grade summaries:

```bash
python -m src.analysis.paper_aggregate \
  --responses-dir raw_responses \
  --manifest data/exam_set.csv \
  --tasks config/tasks.yaml \
  --output-dir results/paper
```

Generate paper figures after aggregation:

```bash
python -m src.analysis.paper_figures
```

Main outputs:

```text
results/
|-- exam_set_table.csv
|-- main_accuracy.csv
|-- calibration.csv
|-- hallucination.csv
|-- parser_audit_sample.csv
|-- graded_all.csv
|-- ablations/
`-- paper/
    |-- coverage_by_model.csv
    |-- headline_by_task.csv
    `-- safety_summary.csv
```

## Reproducibility Notes

- Prompt option order is randomized per item and logged.
- Inference uses deterministic decoding for the paper audit.
- Raw model outputs are stored as JSONL for auditability.
- The parser extracts answer, confidence, abstention, and unsupported findings.
- Missing/unparseable answers are treated as non-coverage.
- Paper headline metrics use neutral MC prompts and exclude degenerate models below the coverage threshold.
- Subject-clustered bootstrap confidence intervals avoid treating correlated slices from the same subject as independent.

## Scope and Responsible Use

This benchmark measures behavioral reliability of VLM outputs on controlled brain-MRI-derived images. It does not validate any model for diagnosis, triage, treatment planning, or autonomous clinical use. Any medical-imaging model intended for clinical workflows requires expert review, prospective validation, regulatory review, and deployment-specific safety monitoring.

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
