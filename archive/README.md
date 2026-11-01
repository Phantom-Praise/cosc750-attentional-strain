# Archived experiments

These files preserve the earlier exploration. Use [the main README](../README.md)
and the root reproduction command for the submission workflow. Historical scripts
may require old evidence files and the full `archive/requirements.txt`.

---

# COSC 750: attentional-strain classification

Stage 3 code for Praise Ben's comparison of classical classifiers on four
**synthetic dataset categories**. The primary models use six usage features.
The proposed refinement is a calibrated RBF-SVM with a validation-selected acute
class decision rule. This is not a clinical diagnostic system or a new SVM algorithm.

## Setup and one-command reproduction

Use **Python 3.12** from this repository directory:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce.py --download-data
```

On Windows, activate with `.venv\Scripts\activate`.

**The final command trains all five baselines and reproduces their table**:
majority reference, histogram Naive Bayes, logistic regression, random forest,
and RBF-SVM. It does not need lecture PDFs or earlier experiment outputs.

The optional download is pinned to Kaggle version 1 and verified by SHA256 before
writing. An existing CSV beside the script is validated and reused. For a file
elsewhere, use `python reproduce.py --data /path/to/short_form_brain_rot_cognitive_decay.csv`.
If the public endpoint is unavailable, obtain version 1 manually as described in
[DATASET.md](DATASET.md). The source repository and ZIP omit the dataset because
its publisher metadata and description give conflicting license labels.

## Results

Outputs go to `results/submission/`:

| Output | Contents |
|---|---|
| `baseline_table.md`, `.csv`, `.tex` | Full five-model comparison; mean and sample SD |
| `baseline_per_run.csv` | Per-seed metrics and timings |
| `baseline_per_class.csv` | Per-class precision/recall/F1 and TP/FP/FN/TN |
| `baseline_comparison.png`, `.svg` | Chart preview and vector figure |
| `baseline/SEED/splits.json` | Training/test and CV row memberships |
| `baseline/SEED/MODEL/` | Predictions, selected settings, full CV results |
| `manifest.json` | Completion state, versions, source/data hashes, hardware |

Existing nonempty output directories are protected. To repeat:

```bash
python reproduce.py --output results/repeat_01
```

Search time includes final refit, which is also reported separately; do not add
it twice. Runtime varies with hardware. Reference outputs in `reference_results/`
are for checking and are never used to generate new scores.

## Reproduce the proposed method too

```bash
python reproduce.py --download-data --proposed --output results/full_reproduction
```

This runs the baselines and a **separate** nested-validation comparison of ordinary
versus calibrated SVM. It generates `proposed_validation_table.md`, `.csv`, `.tex`,
per-fold/per-class reports, and prediction/fold evidence. All folds are regenerated
from fixed seeds; no historical result files are needed.

The proposed model has scaling/SVM pipelines inside three sigmoid calibration
folds, averaged as a calibration ensemble. Inner validation selects C={1,10,100},
gamma={scale,0.03}, and cutoff={None,0.2,0.35,0.5}. None uses probability argmax;
a finite cutoff predicts acute above that probability, otherwise selecting among
the remaining classes. Scaling is refitted inside calibration training subsets.

The equations, algorithm, assumptions, complexity, and permitted claims are in
[docs/PROPOSED_METHOD.md](docs/PROPOSED_METHOD.md).

## Evaluation contract

- 9,000 records; four labels; six numeric usage features. Identifiers, target,
  demographics/context, and cognitive measurements are excluded from main models.
- Baselines: five shared stratified 80/20 splits, seeds 42–46. Each test set has
  1,800 rows including 21 acute rows. Training-only stratified three-fold CV
  selects macro-F1 from 12 configurations per learned model, including NB.
- Proposed comparison: three outer and three inner folds within seed 42's 7,200
  training rows; its original test rows are excluded. Ordinary SVM has 12 settings,
  while the proposed search has 24 plus internal calibration fits.
- Metrics: macro-F1, accuracy, balanced accuracy, and all per-class errors.
  Holdouts overlap; SD is not a confidence interval or evidence of significance.
- The proposed family and later NB addition were explored after earlier results
  were viewed. Preserve the exploratory label. Do not combine proposed-validation
  scores and baseline-test scores into one test-ranking table.

## Source map

| File | Role |
|---|---|
| `reproduce.py` | Portable submission entry point and reporting |
| `run_baselines.py` | Frozen original model definitions/data checks |
| `course_models.py` | Histogram NB and expected-loss utility |
| `research_models.py` | Proposed calibrated SVM and research wrappers |
| `feature_models.py` | Feature groups, ratios, and interaction models |
| `tests/` | Mathematical examples and preprocessing/split/result checks |
| `research/` | Related-code review, provenance, implementation notes |
| `docs/EXPERIMENT_HISTORY.md` | Older experiment commands and prerequisites |

Older `run_*.py` files preserve experimental history. Some depend on saved
historical evidence or lecture files. **Use `reproduce.py` for a fresh clone.**
Failed/interrupted runs, large local results, and model binaries are not included
in the source repository or code ZIP.

## Tests and source package

```bash
python -m unittest discover -s tests -v
python build_submission.py
```

Historical-evidence tests skip when those experiments are absent. Submission tests
check generated default outputs when present. The ZIP includes source, tests,
pinned requirements, documentation, reference tables, and third-party notices.
It excludes the environment, dataset CSV, temporary files, Git history, and models.

## Publication provenance and disclosure

Random forest is the published comparator based on [Breiman (2001)](https://doi.org/10.1023/A:1010933404324).
SVM follows [Cortes and Vapnik (1995)](https://doi.org/10.1007/BF00994018).
We use library implementations, not original authors' software or copied scores.
No strongest published method on this exact dataset was established.

scikit-learn supplies standard learners, preprocessing, calibration, selection,
and metrics. mord and imbalanced-learn support optional research experiments;
notices are in `third_party_licenses/`. Codex generated orchestration, wrappers,
tests, and drafting under user-approved choices; the student must review,
understand, and disclose that work. These synthetic/exploratory findings do not
establish clinical validity, causality, algorithmic novelty, or significant gains.
The paper, slides, actual recording, and repository access remain separate deliverables.
