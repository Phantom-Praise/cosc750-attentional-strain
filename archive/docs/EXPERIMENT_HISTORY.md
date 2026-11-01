# Attentional-strain classification: Stage 3

All code, data, environment files, experiment documentation, and results are in
this `code/` folder. From the parent project folder, run `cd code` before using
the commands below. Earlier run manifests retain their original absolute paths
as historical records; the dataset contents, source hashes, and saved results
are unchanged by the folder move.

For the review of related research and public implementations, see
`research/RELATED_CODE_REVIEW.md`. It records concrete next-experiment suggestions,
source-file revisions, and access limitations; it does not report new model scores.

Reproducible comparison of multinomial logistic regression, RBF-SVM, random forest,
and a majority-class reference on six behavioral features. Read `EXPERIMENT_PLAN.md`
for the prespecified protocol. No clinical interpretation is supported by these
synthetic records.

## Setup (Python 3.12)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Keep `short_form_brain_rot_cognitive_decay.csv` beside the script. Dataset page:
https://www.kaggle.com/datasets/samartalwar/brain-rot-short-form-video-and-attention-span

The runner checks the exact dataset checksum from the experiment plan. Attribution
in the prior paper is S. Talwar; synthetic status and CC BY 4.0 are reported there.
The dataset page's content could not be extracted during this coding session, so
generation details and license still need direct verification before redistribution.

## Reproduce all baseline tables and figures with one command

```bash
python run_baselines.py
```

Default: five seeds, 80/20 stratified holdout per seed, training-only three-fold CV,
12 configurations per learned model. Scaling is inside each training pipeline;
class weights are estimated within the fit. Four search workers use threads;
native numerical-library threads are limited to one. Timings depend on hardware.
The first tied configuration in scikit-learn's deterministic ParameterGrid order wins.

The output directory must be empty. To repeat without overwriting earlier evidence:

```bash
python run_baselines.py --output results/baselines_repeat
```

Outputs in `results/baselines/`:

- `RESULTS.md` and `summary.csv`: mean and sample standard deviation over five runs.
- `per_seed_metrics.csv`: individual scores and timings.
- `manifest.json`: versions, hardware, data/code/protocol hashes, and completion status.
- Each seed: train/test and CV row positions; each model: held-out predictions,
  selected parameters, confusion matrix, and complete CV results where applicable.
- PNG and SVG figures: performance comparison and row-normalized confusion matrices.

Search time includes final refit; do not add refit time to it. Recall SD describes
variation across overlapping holdouts, not independent replications or a confidence
interval. The rarest class contributes 21 records per test set. This is a fixed
baseline study; do not tune features or search spaces against its test results.

## Validation

```bash
python -m unittest discover -s tests -v
```

The tests check dataset schema, participant separation in outer and inner folds,
training-only preprocessing, and saved prediction/metric consistency after the run.

## Lecture-driven extension

Read `COURSE_ALIGNMENT.md` for the lecture-to-code mapping and justified omissions,
and `COURSE_EXTENSION_PLAN.md` for the protocol specified before evaluating the
added histogram Naive Bayes model. Reproduce the extension after the baseline run:

```bash
python run_course_extension.py
```

Outputs go to a separate `results/course_extension/` directory. It must be empty;
use `--output results/course_extension_repeat` to repeat without overwriting.
The extension adds histogram NB, empirical risk, per-class precision/specificity,
ROC/PR curves, and training-only within-class feature correlations. The custom
classifier is in `course_models.py`. Its tests reproduce the teacher's numerical
examples. The original experiment plan, code, and result files remain unchanged.
The extension is exploratory because it follows observation of the original results.

## Exploratory hyperparameter study

The frozen search and validation protocol are in `TUNING_PLAN.md`. To compare the
original SVM search, expanded SVM tuning, and histogram gradient boosting:

```bash
python run_tuning.py
```

This uses nested three-fold CV only within original seed 42's training partition.
The two new searches each evaluate 36 configurations per inner search. Results,
class-wise counts, out-of-fold predictions, candidate lists, split evidence, and
an exploratory fitted model bundle are saved in `results/tuning/`. Use a different
`--output` directory to repeat. The earlier test table is preserved. The new scores
are nested validation scores; do not label them new test-set improvements.

## Research-inspired experiments

The next experiment implements all applicable suggestions from the code review:
ordinal classification, standard/robust/quantile SVM preprocessing, random/SMOTE
minority sampling, sigmoid calibration with a multiclass acute decision rule,
early-stopped boosting, and a balanced forest. Install the updated requirements,
then run:

```bash
python run_research_experiments.py
```

`RESEARCH_EXPERIMENT_PLAN.md` specifies the frozen 24-candidate searches and saved
nested folds. Results and fold-specific models go to `results/research_experiments/`.
Use `--output` with a new empty directory for a repeat. The original SVM reference
is reused from its existing out-of-fold predictions; its zero search time in this
new report reflects reuse, not instantaneous training. No original test rows are
evaluated. `research_models.py` contains the estimator wrappers.

The ordinal objective/gradient and probability routines are reused from mord 0.7;
sampling and balanced forests are reused from imbalanced-learn 0.14.0. Versions
are pinned and their notices are retained under `third_party_licenses/`. Our
wrappers, validation orchestration, checks, and reports were generated with Codex.
The method review is provenance, not a claim of clinical validity or novelty.

## Feature additions and interactions

Run the frozen feature study after installing the requirements:

```bash
python run_feature_ablation.py
```

`FEATURE_ABLATION_PLAN.md` defines the grouped additions, engineered proxies,
logarithms, interaction expansion, and matched 12-setting searches. The new
estimators are in `feature_models.py`. Outputs, selected feature names, class-wise
counts, and fold models are saved in `results/feature_ablation/`. Use `--output`
with another empty directory for a repeat. Context and demographic/platform
variants use broader inputs and must be distinguished from usage-only models.
No cognitive outcome measurements enter any of these variants.

## Method provenance and remaining submission work

The orchestration, audit, and plotting code were generated with Codex and must be
reviewed and disclosed by the student. Learners, preprocessing, CV, and metrics are
reused from scikit-learn, not reimplementations of the original algorithms.

- Cortes and Vapnik (1995), Support-vector networks: https://doi.org/10.1007/BF00994018
- Breiman (2001), Random Forests: https://doi.org/10.1023/A:1010933404324
- scikit-learn grid search: https://scikit-learn.org/1.7/modules/generated/sklearn.model_selection.GridSearchCV.html

These are published method families. This run does not establish a strongest
published method on this specific dataset or reproduce a domain paper's full
pipeline. Resolve that rubric requirement and specify a proposed method before
claiming the complete Stage 3 submission is ready. A hosted Git repository, paper,
extended slides, and actual recorded talk are still separate deliverables.
