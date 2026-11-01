# Feature study interpretation

The frozen plan is ../FEATURE_ABLATION_PLAN.md; the reproducible runner is
../run_feature_ablation.py. Results use the same three nested outer folds within
the same 7,200 original training rows. No original test rows were evaluated.

## Observed findings

- Adding sleep and multitasking raised mean macro-F1 from 0.8462 to 0.8499, a
  difference of about 0.37 percentage points. Mean acute precision fell from
  0.8429 to 0.8274 and recall from 0.7024 to 0.6905. This is a small broader-input
  result, not a demonstration of improved acute detection or real-world validity.
- Adding age, gender, and platform jointly produced mean macro-F1 0.8363, below
  the six-feature SVM. This does not establish that every added variable is
  individually irrelevant; they were evaluated as a group with a fixed search.
- Four engineered ratios/interactions gave macro-F1 0.8416. Adding six log1p
  representations gave 0.8394. Acute precision rose slightly, while recall fell.
  Those variants do not improve both precision and recall over the reference.
- Interaction logistic regression gave macro-F1 0.8411 versus 0.8443 for its
  matched linear logistic control. The extra terms did not improve this metric.
- The linear logistic control's mean acute recall was 0.7381, higher than the
  SVM's 0.7024, but precision was lower (0.8084 versus 0.8429).

Across pooled OOF acute predictions, the context variant found 58 of 84 acute
records with 14 false alarms, versus 59 found and 11 false alarms for the original
SVM. The logarithmic engineered variant found 55 with 9 false alarms, illustrating
the precision/recall trade-off. Mean-fold precision and pooled precision differ
because the number of positive predictions varies by fold.

The previously studied calibrated six-feature SVM had mean macro-F1 0.8492 on
these same folds. Its difference from the broader-context score is tiny and does
not justify declaring either method reliably superior. Keep usage-only and
broader-input results separate. No numerical performance ceiling was established.

## Input and interpretation checks

Allowed raw inputs were limited to the original six plus sleep, multitasking,
age, gender, and platform. Identifier, target, and all six excluded cognitive
measurements never entered feature construction or preprocessing. Whether any
additional inputs share a synthetic generation formula with the tier remains
unverified. Sleep/multitasking may require self-report and should not be described
as passively observed phone-usage logs.

Training-only ratio auditing found 18 of 7,200 records with bedtime scrolling
minutes greater than total daily short-form minutes. No training record had
short-form hours above total screen hours. The study retained the values and
reported bedtime shares above 1; it did not clamp ratios, delete rows, or change
partitions after evaluation. This is a concrete synthetic-data consistency limit.
Derived predictors are proxies, not independently measured session durations or
attention outcomes.

Fitted dimensions were 8 for context, 14 after demographic/platform encoding,
10 for engineered features, 16 with logs, 6 for linear logistic, and 21 for main
effects plus pairwise logistic interactions. Selected parameters and exact
feature names are saved for each fold. Scaling and encoding are fitted on the
current training fold only; unseen categories are ignored at transformation.

All 23 project tests pass, including feature formulas, prohibited-column
exclusion, training-only preprocessing, matched search grids, exact saved folds,
test-row exclusion, and recomputation of saved OOF scores. No extra packages were
needed for this study. The source, validation orchestration, tests, and reports
were generated with Codex and should enter the AI-use disclosure.

These results are exploratory: earlier outcomes informed the study, folds have
overlapping training sets, and many method families have now been compared.
Fold SD is not a confidence interval. The new results are useful ablation
evidence for the paper, not an independently confirmed improvement claim.
