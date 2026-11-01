# Feature ablation and engineering: exploratory protocol

Specified before fitting these feature variants. Prior test and validation results
motivated this study. It remains exploratory; no new untouched test is claimed.

Reuse exactly the original seed 42 training rows and saved outer/inner nested
folds from results/tuning. No original test rows are fitted, selected on, or scored.
Keep the original SVM's saved OOF predictions as the paired six-feature reference.

## Variants

- svm_context: original six plus total_sleep_hours and multitasking_score. This
  uses broader context, potentially requiring self-report, not usage logs alone.
- svm_demographics_platform: original six plus age, gender, primary_platform.
  Fit categorical one-hot encoding within each training fold; ignore unknown
  validation categories. This tests the group jointly, not each column separately.
- svm_engineered: original six plus four deterministic proxies:
  short-form minutes/session = 60*daily hours / sessions;
  swipe-clip ratio = swipes/minute * clip seconds / 60;
  bedtime share = bedtime minutes / (60*daily hours);
  short-form screen share = daily short-form hours / total screen hours.
  Denominators must be positive. Ratios are not clipped to [0,1]; report any
  inconsistent raw measurements rather than silently fixing them.
- svm_engineered_log: the previous ten columns plus log1p of each original
  nonnegative feature, yielding 16 predictors. This isolates adding logarithmic
  representations from adding ratios alone.
- logistic_linear: original six, standard-scaled, with L2 multinomial logistic
  regression. A matched control for the interaction model.
- logistic_interactions: original six, standardized inside fitting, expanded to
  the six main effects plus 15 pairwise products (no squared terms), standardized
  again and fit with the same regularized logistic search. No engineered ratios,
  context, or categorical columns enter this model.

All four new SVM variants use exactly the original 12-configuration grid:
C={0.1,1,10}; gamma={scale,0.1}; class_weight={None,balanced}. The two logistic
variants each use C={0.01,0.1,1,10,100,1000} crossed with class_weight={None,balanced}
(12 configurations), lbfgs, max_iter=5000. This holds the SVM search fixed while
features change and holds logistic search fixed while interactions change.

The only permitted raw predictors across all variants are the original six,
sleep, multitasking, age, gender, and platform. Exclude identifiers, the target,
brain fog, attention retention, sustained attention, working memory, task-switch
latency, and burnout index. Their exact relationship to target generation remains
unverified; their exclusion preserves the stated prediction setting.

## Evaluation

Select mean inner macro-F1, refit on outer training data and predict outer
validation once. Report mean ± sample fold SD of macro-F1, accuracy, balanced
accuracy, acute precision/recall, and runtime; save all class-wise pooled OOF
precision/recall/F1 and TP/FP/FN/TN for 7,200 unique rows. Report feature names
after encoding/expansion and selected settings. Fold SD is not a confidence
interval. Previously studied calibration can be shown as historical context,
but do not merge it into these feature variants without a separate experiment.

These are grouped additions to a common base, not causal feature effects or a
complete feature importance study. Demographic/platform effects and all broader-
input improvements must be reported separately from usage-only improvements.
No deployed final model or new test evaluation is produced automatically.
