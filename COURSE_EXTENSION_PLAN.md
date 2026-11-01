# Lecture-driven extension, October 26, 2026

Specified after viewing the original four-model results and reading the four
teacher-provided lecture PDFs, but before evaluating the new histogram model.
This is an exploratory extension, not a newly sealed confirmatory test.

## Frozen comparison

Keep `run_baselines.py`, `EXPERIMENT_PLAN.md`, and `results/baselines/` unchanged.
Use its saved outer splits and inner folds for a new histogram Naive Bayes baseline.
Primary features and all five seeds stay identical. Select bin count in
{3, 5, 10, 15, 20, 30} and smoothing alpha in {0.5, 1.0} by training-only
three-fold macro-F1: 12 configurations, matching the original learned models' budget.
Alpha=1 is the lecture's add-one smoothing; alpha=.5 is another fixed symmetric
Dirichlet smoothing choice. Do not select either using test results.

Fit shared equal-width edges per feature using only the current training fold,
then count each class separately. Estimate class priors by training frequencies.
Use `(count + alpha)/(class_count + alpha * number_of_bins)` for bin probability,
and divide by bin width for density. Multiply the densities and class priors in
log space, then normalize with log-sum-exp. For a feature outside its fitted range,
clip its value to the nearest boundary bin; record this policy explicitly. This
models a clipped feature representation, not a justified tail density.

The point prediction minimizes expected zero-one loss under the estimated
posterior. The implementation accepts a documented general loss matrix, but the
experiment uses zero-one loss throughout. No arbitrary clinical error costs are
introduced. These estimated probabilities are not assumed calibrated or equal to
the unknown true posterior. Naive Bayes assumes conditional feature independence;
assess within-class correlations using training rows only as a diagnostic.

## Evaluation additions

Compute per-class precision, recall, F1, specificity, false-positive rate, and
empirical zero-one risk from existing held-out predictions, plus the new model.
For each original model, refit its already-selected parameters on its original
training rows solely to recover held-out ranking scores. Require exact agreement
with saved predictions. No new tuning and no replacement of original scores.
Use SVM decision-function outputs as ranking scores, not as probabilities.

Compute one-vs-rest ROC-AUC and average precision per class and per seed. Plot
acute-versus-rest ROC and precision-recall curves for prespecified seed 42, and
label it as a single split. Save all five seeds' metrics. Do not choose a threshold
from the test ROC or PR curve. Report per-split Wilson intervals for acute recall;
do not pool overlapping holdouts as independent observations.

## Verification

Reproduce the histogram lecture's worked example, including the smoothed
(90, 62) posterior of 0.4 Fail and 0.6 Pass; test density integrals, posterior
normalization, boundary handling, and a known unequal-cost decision. Verify the
extension uses the saved folds and agrees with original prediction evidence.
Create a topic-by-topic coverage report explaining applicable omissions.
