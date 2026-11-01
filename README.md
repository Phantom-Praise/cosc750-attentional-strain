# Attentional-Strain Classification

**COSC 750 · Stage 3 · Praise Ben**

A reproducible comparison of five classical classifiers using six short-form-video
usage features to predict four **synthetic attentional-strain categories**.

## Start here

| I want to… | Open |
|---|---|
| See the results | [Baseline table](reference_results/baseline_table.md) |
| Run the project | [Quick start below](#quick-start) |
| Understand the models | [Method](docs/PROPOSED_METHOD.md) |
| Find the code | [`strain/`](strain/) |
| Read supporting details | [Documentation](docs/README.md) |

## Results

Mean macro-F1 ± sample SD across five held-out splits:

| Model | Macro-F1 |
|---|---:|
| Majority reference | 15.02% ± 0.00 |
| Histogram Naive Bayes | 67.65% ± 1.28 |
| Logistic regression | 83.27% ± 2.28 |
| Random forest | 82.91% ± 2.48 |
| **RBF-SVM** | **83.87% ± 2.07** |

The SVM reached **87.04% accuracy**. A calibrated-SVM refinement is evaluated in a
[separate exploratory validation study](reference_results/proposed_validation_table.md).
These results concern synthetic data, not clinical diagnosis.

## Quick start

Python **3.12**:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce.py --download-data
```

This downloads and verifies dataset version 1, retrains all five baselines, and
writes tables, charts, and prediction evidence to `results/submission/`.

Add `--proposed` to reproduce the calibrated-SVM comparison too. To repeat without
overwriting results, add `--output results/repeat_01`.

[More running options](docs/RUNNING.md) · [Dataset](docs/DATASET.md) · [Verification](docs/SUBMISSION_VERIFICATION.md)

## Project layout

```text
reproduce.py        Run the study
strain/             Data checks, models, evaluation
reference_results/  Recorded tables and charts
tests/             Verification checks
docs/              Method and supporting documentation
tools/             Code-package builder
archive/           Earlier exploratory work
```

The main workflow does not require the archive. All results are calculated from
newly fitted models; recorded tables are provided only for comparison.
