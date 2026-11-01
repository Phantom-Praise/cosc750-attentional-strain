# Research-inspired implementation notes

All applicable recommendations in RELATED_CODE_REVIEW.md have now been implemented:

| Suggestion | Implementation |
|---|---|
| Ordered categories | OrdinalLogistic in research_models.py, using mord's immediate-threshold objective/gradient and ordered cutpoints. |
| Representation comparison | PreprocessedSVM selects standard, robust, or normal-quantile transformation within the inner search. |
| Controlled oversampling | OversampledSVM scales and then applies random oversampling or SMOTE within fitting only, with acute counts calculated from that training fold. |
| Probability calibration and decision rule | CalibratedAcuteSVM fits a sigmoid calibration ensemble with preprocessing inside the calibration folds. Cutoff selection is nested within outer validation. |
| Longer, shallow, early-stopped boosting | EarlyStoppedBoosting uses a 1,000-iteration ceiling and a training-internal stopping split. Actual selected iterations are saved. |
| Balanced ensembles | BalancedRandomForestClassifier is tested independently of oversampling and class weights. |

Each new search evaluates 24 configurations over the same three saved inner
folds for each of the same three saved outer folds. There are 1,296 inner
candidate fits plus 18 outer refits; calibration internally trains additional
base estimators. Thus equal configuration budgets do not imply equal wall-clock
or base-fit budgets. The original SVM reference is reused exactly from its saved
predictions, not retrained using new test information.

## Provenance and reuse

- mord 0.7 provides obj_margin, grad_margin, threshold_predict, and threshold_proba.
  It is credited to Fabian Pedregosa and contributors. The wrapper adds explicit
  four-tier encoding, scaling, modern estimator integration, and convergence checks.
- imbalanced-learn 0.14.0 supplies random oversampling, SMOTE, the sampling pipeline,
  and balanced random forests.
- scikit-learn supplies SVM, transforms, calibration, boosting, CV, and metrics.
- Upstream license notices are preserved in ../third_party_licenses/. The package
  versions are pinned in ../requirements.txt.
- Codex generated the wrappers, experiment runner, tests, and reports. This reuse
  and AI contribution should be added to the paper's disclosure.

The first completed run produced feature-name warnings because the ordinal fit
converted a DataFrame into a numeric array, while prediction passed a DataFrame
to its scaler. The values and feature order were unaffected. Prediction now
validates and converts inputs consistently. The earlier run is retained as
results/research_experiments_input_warning_run, and the full frozen comparison
was rerun with the corrected input handling. Candidate lists and statistical
choices were unchanged. Final verification compares held-out predictions between
both runs to establish that the format correction did not change decisions.

## Interpretation

The trial grid includes an ordinary calibrated argmax rule as well as acute
cutoffs. A selection of cutoff=None means validation preferred calibrated
multiclass argmax; it does not mean a special acute threshold was applied.
Probabilities remain model estimates, not clinical risk estimates.

Ordinal threshold decisions follow mord's immediate-threshold classifier. Their
predict_proba method exposes probabilities from cumulative differences, but
prediction is not redefined as probability argmax. The surrogate does not optimize
macro-F1 directly; inner validation selects alpha and class weights using macro-F1.

SMOTE may create fractional session counts. This is an explicit approximation in
a synthetic-data benchmark. Original validation rows are never resampled. Balanced
forests trade majority-class information for a more balanced per-tree sample;
higher acute recall is not proof of better overall decisions.

All findings are exploratory. The original seed 42 test rows are excluded, but
earlier test and validation results informed this research direction. None of
the fold-specific models is automatically deployed or promoted to a final model.
The social-media sequence model remains infeasible without ordered app events.

## Verified outcome

All seven procedures reproduced identical held-out predictions after the ordinal
input-format correction. All 18 project tests pass, including saved-fold identity,
test-row exclusion, calibration preprocessing separation, sampling behavior,
ordinal convergence/order, and recomputation of saved metrics.

The calibrated SVM procedure has the highest fold-mean macro-F1 (0.8492 versus
0.8462 for the original SVM), with mean acute precision 0.8551 versus 0.8429 and
identical mean acute recall 0.7024. The pooled acute counts are TP=59, FP=10,
FN=25, TN=7106, compared with TP=59, FP=11, FN=25, TN=7105 for the reference.
This is a small exploratory gain, not evidence of a reliable improvement in both
precision and recall. Calibration and cutoff selection were evaluated jointly,
so this comparison alone does not isolate either component's causal effect.

The balanced forest recovers 76 of 84 acute validation records but generates 73
acute false alarms. Ordinal classification improves mean acute recall but has
lower mean precision and macro-F1 than the reference. No family uniformly
dominates all metrics. The original test table remains unchanged.
