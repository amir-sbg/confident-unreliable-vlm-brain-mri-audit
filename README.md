# Confident but Unreliable: VLM Brain MRI Safety Audit

Research code for **"Confident but Unreliable: A Behavioral Safety Audit of Vision-Language Models on Brain MRI"**, accepted to [ACM AI 2026](https://aisummit.acm.org/) after submission in **2026.05** and acceptance in **2026.07**.

This project was developed in the TReNDS Center research environment as part of a Georgia Tech and Georgia State University collaboration. It audits whether vision-language models can recognize basic brain-MRI properties while also communicating uncertainty safely. The main point is deliberately behavioral: a model can answer almost every question, sound confident, and still be poorly calibrated when it is wrong.

This is an inference-only evaluation framework. It does not train or fine-tune model weights, and it is not a clinical diagnostic system.

<p align="center">
  <img src="docs/accuracy-vs-error-confidence.png" alt="Accuracy versus confidence on wrong answers" width="620">
</p>

## Paper Snapshot

The accepted paper evaluates six instruction-tuned VLMs on **4,102 images**: 4,032 axial, coronal, and sagittal MRI slices from 250 subjects, plus 70 non-brain and noise controls. Labels are derived automatically from public metadata, slice geometry, and released expert segmentation masks, avoiding new manual annotation.

The audit reports accuracy together with safety-relevant behavior:

- coverage and answered-item accuracy
- balanced accuracy
- expected calibration error and Brier score
- mean stated confidence on incorrect answers
- confidently-wrong rate
- open-ended hallucination rate
- abstention on non-brain controls

The headline result is that accuracy and confidence reliability move separately. Gemma-4-12B is the most accurate model in the pilot study, but it also gives the highest mean confidence on its wrong answers. Qwen2.5-VL-7B is less accurate overall but substantially less overconfident when wrong.

| Model | Coverage | Accuracy | ECE | Brier | Confidence on wrong answers | Confidently-wrong rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| InternVL2.5-8B | 1.000 | 0.567 | 0.361 | 0.371 | 0.917 | 0.426 |
| Qwen2.5-VL-3B | 1.000 | 0.514 | 0.361 | 0.355 | 0.844 | 0.357 |
| Qwen2.5-VL-7B | 1.000 | 0.563 | 0.272 | 0.326 | 0.819 | 0.360 |
| Gemma-3-4B | 1.000 | 0.531 | 0.404 | 0.405 | 0.922 | 0.462 |
| Gemma-4-12B | 0.996 | 0.670 | 0.307 | 0.311 | 0.968 | 0.330 |
| MedGemma-4B | 1.000 | 0.571 | 0.381 | 0.389 | 0.949 | 0.428 |

Lower is better for ECE, Brier, confidence-on-wrong, and confidently-wrong rate.

## What the Code Does

The pipeline turns public neuroimaging volumes into slice-level VLM audit tasks:

1. Extract controlled 2D slices from NIfTI volumes.
2. Build a manifest with labels from acquisition metadata, extraction geometry, and segmentation masks.
3. Render deterministic multiple-choice and open-ended prompts.
4. Run local or API-based VLM inference with resumable JSONL outputs.
5. Parse answers, confidence values, findings, refusals, and abstentions.
6. Grade responses against slice-level labels.
7. Aggregate calibration, confident-error, hallucination, and abstention metrics.
8. Generate paper-style reliability figures.

## Evaluation Scope

The paper focuses on five automatically gradable brain-MRI tasks:

| Task | Question |
| --- | --- |
| `T1-MOD` | Which MRI modality or sequence is shown? |
| `T2-PLANE` | Is the slice axial, sagittal, or coronal? |
| `T3-ISBRAIN` | Is the image a brain MRI? |
| `T4-TUMOR` | Does the image show a visible tumor or abnormality? |
| `T5-LAT` | Where is the abnormality located: left, right, bilateral, or none? |

The repository also contains configurable VQA and abstention tracks for broader experiments. Multiple-choice options are shuffled deterministically and logged, and every prompt asks the model to state a confidence from 0 to 100 when possible.

## Data and Labeling

The code supports:

- **BraTS 2024** post-treatment glioma volumes and segmentation masks
- **IXI** healthy-volunteer scans
- **OASIS** T1 scans as an optional configured healthy dataset
- generated and non-brain negative controls for out-of-distribution testing

Slice labels are derived without new annotation:

- sequence from acquisition metadata
- plane from extraction geometry
- brain/non-brain status from source
- tumor presence from the in-slice segmentation mask
- laterality from the lesion centroid relative to the mid-sagittal line

The paper uses a visibility threshold of at least 25 lesion voxels for tumor-positive slice labels. This matters because many slices from a tumor-bearing subject contain no visible tumor.

## Model Backends

Model definitions live in `config/models.yaml`. The inference layer supports:

- local vLLM runs for supported open VLMs
- Hugging Face Transformers fallback
- optional API-backed runners
- resumable output streams keyed by model, image, and prompt
- SLURM launchers for cluster sweeps

The audited panel includes InternVL2.5-8B, Qwen2.5-VL-3B/7B, Gemma-3-4B, Gemma-4-12B, and MedGemma-4B.

## Quick Start

The project targets Python 3.11 with PyTorch, CUDA-compatible inference libraries, Hugging Face tooling, neuroimaging packages, and the analysis dependencies listed in `environment.yml` and `requirements.txt`.

```bash
git clone https://github.com/amir-sbg/vlm-brain-mri-audit.git
cd vlm-brain-mri-audit

conda env create -f environment.yml
conda activate vlmaudit
export PYTHONPATH="$PWD"

python -m pytest tests/test_parse.py -q
```

Check configured datasets:

```bash
python -m src.data.download \
  --check-only \
  --dataset brats ixi oasis
```

Build slices and the exam manifest after source data is available:

```bash
python -m src.data.slice_extract \
  --config config/datasets.yaml \
  --dataset brats ixi oasis negative_controls \
  --output-dir data/slices \
  --provenance-out data/slice_provenance.jsonl

python -m src.data.build_manifest \
  --provenance data/slice_provenance.jsonl \
  --config config/datasets.yaml \
  --output data/exam_set.csv
```

Run inference for one configured model:

```bash
python -m src.inference.run_inference \
  --models config/models.yaml \
  --tasks config/tasks.yaml \
  --prompts config/prompts.yaml \
  --manifest data/exam_set.csv \
  --output-dir raw_responses \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --phrasing neutral \
  --format MC OE \
  --batch-size 8
```

Aggregate responses and generate figures:

```bash
python -m src.analysis.aggregate \
  --responses-dir raw_responses \
  --manifest data/exam_set.csv \
  --tasks config/tasks.yaml \
  --output-dir results

python -m src.analysis.make_figures \
  --results-dir results \
  --output-dir figures \
  --manifest data/exam_set.csv
```

For the cluster workflow, review account, partition, node, paths, and conda location before submitting:

```bash
sbatch slurm/run_all.slurm
```

## Engineering Notes

- Dataset sampling and negative-control generation are seeded.
- Prompt templates, answer spaces, model settings, and dataset paths are configuration-driven.
- Multiple-choice option order is deterministic and recorded for grading.
- Each model writes independent JSONL outputs, so interrupted workers can resume without duplicating completed items.
- Parser tests cover answer extraction, confidence parsing, and refusal handling.
- Raw data, model caches, responses, generated figures, logs, W&B runs, and internal collaboration notes are excluded from Git.

## Repository Structure

```text
.
├── config/
│   ├── datasets.yaml
│   ├── models.yaml
│   ├── prompts.yaml
│   └── tasks.yaml
├── docs/
│   └── accuracy-vs-error-confidence.png
├── slurm/
├── src/
│   ├── analysis/
│   ├── data/
│   ├── inference/
│   ├── prompts/
│   ├── scoring/
│   └── utils/
├── tests/
├── environment.yml
└── requirements.txt
```

## Suggested Repository Metadata

**Name:** `confident-unreliable-vlm-brain-mri-audit`

**About:** Accepted ACM AI 2026 safety-audit framework for vision-language models on brain MRI, measuring calibration, confident errors, hallucination, abstention, and slice-level accuracy.
